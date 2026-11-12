"""学习接口插件自动发现 + 热加载监视。

约定：
  * 插件目录 vis/learning_plugins/*.py（跳过 _ 开头与 __init__）
  * 每个模块导出 PLUGIN = {id, name, description?, params?, create}
  * create(params) 返回带 command(state)->(w1,w2) 的对象
  * 加载失败的文件不会出现在选项栏，错误以字符串返回给宿主显示
"""
import glob
import importlib.util
import os
import sys

from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer, Signal

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.join(VIS_DIR, "learning_plugins")


def plugin_dir():
    return PLUGIN_DIR


def _ensure_sys_path():
    if VIS_DIR not in sys.path:
        sys.path.insert(0, VIS_DIR)


def signature(dirpath=None):
    """文件名 -> (mtime_ns, size)，用于轮询兜底与变更判定。"""
    d = dirpath or PLUGIN_DIR
    out = {}
    for path in glob.glob(os.path.join(d, "*.py")):
        try:
            st = os.stat(path)
        except OSError:
            continue
        out[os.path.basename(path)] = (st.st_mtime_ns, st.st_size)
    return out


def _normalize_params(raw, pid, errors):
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        errors.append("%s: params 必须是列表" % pid)
        return []
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append("%s: params[%d] 不是字典" % (pid, i))
            continue
        key, typ = item.get("key"), item.get("type", "float")
        if not key or typ not in ("float", "bool", "choice"):
            errors.append("%s: params[%d] 缺 key 或 type 非法" % (pid, i))
            continue
        spec = {"key": str(key), "label": str(item.get("label", key)), "type": typ}
        if typ == "float":
            spec.update({
                "min": float(item.get("min", 0.0)),
                "max": float(item.get("max", 1.0)),
                "step": float(item.get("step", 0.05)),
                "default": float(item.get("default", item.get("min", 0.0))),
            })
        elif typ == "bool":
            spec["default"] = bool(item.get("default", False))
        else:
            spec["choices"] = [str(x) for x in item.get("choices", [])]
            spec["default"] = str(item.get("default", ""))
        out.append(spec)
    return out


def discover(dirpath=None):
    """扫描插件目录，返回 (plugins, errors)。每次调用都重新加载模块。"""
    d = dirpath or PLUGIN_DIR
    _ensure_sys_path()
    plugins, errors = [], []
    os.makedirs(d, exist_ok=True)

    for path in sorted(glob.glob(os.path.join(d, "*.py"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("_"):
            continue
        try:
            st = os.stat(path)
            mod_name = "_learning_plugin_%s_%d" % (stem, st.st_mtime_ns)
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                errors.append("%s: 无法创建模块 spec" % stem)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(mod_name, None)

            meta = getattr(module, "PLUGIN", None)
            if not isinstance(meta, dict):
                errors.append("%s: 缺少 PLUGIN 字典" % stem)
                continue
            pid = str(meta.get("id", stem))
            name = str(meta.get("name", pid))
            create = meta.get("create")
            if not callable(create):
                errors.append("%s: create 不可调用" % pid)
                continue
            plugins.append({
                "id": pid,
                "name": name,
                "description": str(meta.get("description", "")),
                "params": _normalize_params(meta.get("params"), pid, errors),
                "create": create,
                "path": path,
            })
        except Exception as exc:  # 单个插件失败不影响其它插件
            errors.append("%s: %s: %s" % (stem, type(exc).__name__, exc))

    plugins.sort(key=lambda p: p["id"])
    return plugins, errors


class LearningWatcher(QObject):
    """目录变更即时感知：QFileSystemWatcher + 轮询兜底 + 防抖。

    Windows 上编辑器保存常触发多次 fileChanged，且 Qt 会移除被替换文件的
    监视路径，所以每次发信号后重新注册文件。
    """

    changed = Signal()

    def __init__(self, dirpath=None, parent=None, poll_ms=1200):
        super().__init__(parent)
        self.dirpath = dirpath or PLUGIN_DIR
        self._sig = signature(self.dirpath)

        self._fs = QFileSystemWatcher(self)
        self._fs.addPath(self.dirpath)
        self._rewatch_files()
        self._fs.directoryChanged.connect(self._schedule)
        self._fs.fileChanged.connect(self._schedule)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self._emit)

        self._poll = QTimer(self)
        self._poll.setInterval(poll_ms)
        self._poll.timeout.connect(self._poll_check)
        self._poll.start()

    def _rewatch_files(self):
        watched = set(self._fs.files())
        for path in glob.glob(os.path.join(self.dirpath, "*.py")):
            if path not in watched:
                self._fs.addPath(path)

    def _schedule(self, *_args):
        self._debounce.start()

    def _poll_check(self):
        sig = signature(self.dirpath)
        if sig != self._sig:
            self._sig = sig
            self._schedule()

    def _emit(self):
        self._sig = signature(self.dirpath)
        self._rewatch_files()
        self.changed.emit()

# `.neuro` 容器头 · 定稿 v1（M0 冻结）

> 依据：设计定稿 §9.1 / §9.2；实现：`core/container_neuro.h`
> 验收：`build_m0.ps1`

## 1. 头部布局（72 B，小端，packed）

| 偏移 | 字段 | 类型 | 值 / 说明 |
|---|---|---|---|
| 0 | `magic[8]` | u8×8 | `NEURO\0\0\1` |
| 8 | `bom` | u8 | `0x01` = 小端 |
| 9 | `float_format` | u8 | `0x01` = IEEE 754 |
| 10 | `format_version` | u16 | 当前 1 |
| 12 | `min_runner_version` | u16 | 当前 1 |
| 14 | `header_size` | u16 | 必须等于 72 |
| 16 | `seed.structure` | u64 | 结构初始化流 |
| 24 | `seed.weight` | u64 | 权重初始化流 |
| 32 | `seed.noise` | u64 | 噪声流 |
| 40 | `seed.trial_order` | u64 | 试次顺序流 |
| 48 | `config_digest[8]` | u8×8 | 配置摘要（参数+宏+编译选项） |
| 56 | `index_offset` | u64 | 区块表起始偏移，≥ 72 |
| 64 | `block_count` | u32 | 1..9 |
| 68 | `reserved` | u32 | 0 |

## 2. 区块表项（48 B）

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | u32 | 见 §3 |
| `flags` | u32 | HOT / COMPRESSED / ZERO_COPY |
| `offset` | u64 | 文件绝对偏移 |
| `comp_size` | u64 | 磁盘字节数 |
| `raw_size` | u64 | 解压后字节数 |
| `count` | u64 | 元素数（不透明块为 0） |
| `elem_size` | u32 | 单元素字节（不透明块为 0） |
| `crc32` | u32 | 压缩后字节的 CRC32 |

## 3. 区块类型

| 值 | 类型 | 冷热 | 内容 |
|---|---|---|---|
| 1 | `string_pool` | 冷 | ID 字符串、`display_name` |
| 2 | `metadata` | 冷 | 能力向量、统计摘要、版本 |
| 3 | `weight` | 热 | float32 权重数组 |
| 4 | `eligibility_trace` | 热 | float32 ×3/突触 |
| 5 | `metaplasticity` | 热 | float16/突触，与迹连续 |
| 6 | `memory_slots` | 冷 | 痕迹槽签名 |
| 7 | `decoder_state` | 冷 | PDc 权重/后验 |
| 8 | `snc_records` | 冷 | 分化/竞争/修剪记录 |
| 9 | `primitive_library` | 冷 | PPm 基元与链权重 |

## 4. 校验顺序（强制）

```
1. 读前 10 字节 → neuro_magic_probe()      # 只比较 magic/bom/float
2. 读完整 72 字节 → neuro_header_validate() # header_size/block_count/index_offset
3. neuro_header_check_runner(h, runner)     # 版本门控，双向拒绝
4. 之后才允许 mmap 并强转热数据指针
```

- 版本规则：`runner < min_runner_version` → `E_RUNNER_TOO_OLD`；`format_version > 支持版本` → `E_FORMAT_TOO_NEW`。**不降级读取、不尽力解析。**
- `neuro_magic_probe()` 可在只有 10 字节可用时调用（网络/管道头）。

## 5. 热数据布局

`SynapseHot`（32 B）：

| 偏移 | 字段 | 类型 |
|---|---|---|
| 0 | `pre` | u32 |
| 4 | `post` | u32 |
| 8 | `w` | f32 |
| 12 | `e[3]` | f32×3 |
| 24 | `p_half` | f16 |
| 26 | `flags` | u16 |
| 28 | `last_tick` | u32 |

- 600 万突触 ≈ 192 MB；5000 万 ≈ 1.6 GB。
- `p_half` 在更新时提升为 float32，落盘保持 f16。
- `last_tick` 用于惰性衰减：结算时才按 `Δt` 批量衰减 `w/e`，不做全局扫描。

## 6. 检查点、增量与回滚

- 检查点：临时文件 → `fsync` → `rename` 原子替换。
- `.delta` 记录 `{tick, block_type, offset, old_value, new_value}`；回滚 = 最近检查点 + 反向应用增量。
- **资格迹不回滚。** 回滚后必须对迹做时间轴结算，否则旧信用会与新权重错配（幽灵更新）。v1 规则：
  - 已知回滚间隔 `Δticks` 且单 tick 衰减系数 `λ`：`e *= λ^Δticks`；
  - 间隔未知或 `Δticks > 5τe`：直接清零。
  - 实现：`neuro_trace_after_rollback(e, decay_per_tick, delta_ticks)`。
- 回滚事件写 `health.jsonl`，字段含 `checkpoint_id`、`delta_count`、`trace_action`（decayed/cleared）。

## 7. 遗留

- Zstd 冷数据分块（M5 落地，当前只定义字段）；
- CRC32 校验（M5）；
- `config_digest` 的计算范围（参数+宏+编译器版本）在 M5 定稿；
- 跨进程共享内存只读映射（`neuro-play`）在 M5 验证。

## 8. 变更规则

改字段/枚举/热布局必须：升 `format_version` + 更新 `min_runner_version` + 提供迁移工具 + 在 `build_info.json` 记录新哈希。禁止静默修改。

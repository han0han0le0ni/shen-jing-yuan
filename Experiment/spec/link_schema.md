# Link 端口契约 · 草案 v1（M0）

> 依据：设计定稿 §2.3；本机暂无 `flatc`，先用字段表冻结语义，M5 再落 `.fbs` 并生成 C/Python 绑定。
> 所有消息带 `schema_version`；新增字段只能追加到尾部并保留默认值。

## 1. 消息信封

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | u16 | 当前 1 |
| `msg_type` | u16 | 见 §2 |
| `tick` | u64 | 发送方视角的 tick |
| `payload` | bytes | 对应消息体 |

## 2. 端口与消息体

### BRAIN.IN（Body → Brain）
| 字段 | 类型 | 说明 |
|---|---|---|
| `modality` | u8 | 1 文本 2 图像 3 听觉 4 关节 5 目标位置 |
| `feature_ref` | u64 | 共享内存中的特征块句柄 |
| `feature_len` | u32 | 元素数 |
| `context_flags` | u32 | bit0 执行期、bit1 试次间、bit2 睡眠 |

约束：不得携带奖励；不得携带权重。

### BRAIN.IN.MODULATION（Body → Brain）
| 字段 | 类型 | 说明 |
|---|---|---|
| `m_rw` | f32 | 奖励（可负） |
| `m_ac` | f32 | 新颖性（已滤波或原始，由 `ac_filtered` 标志区分） |
| `m_er` | f32×4 | 误差向量（关节/末端维度，未用填 0） |
| `er_dim` | u8 | 有效维度 1–4 |
| `reward_valid` | u8 | 0 时整条奖励丢弃，基线不更新 |
| `source` | u8 | 1 用户 2 外部 AI 3 内部误差 |

### BRAIN.OUT.PROPOSE（Brain → Body）
| 字段 | 类型 | 说明 |
|---|---|---|
| `intent_id` | u64 | 与意图缓冲对应 |
| `type` | u8 | 1 文本 2 动作 3 情绪 |
| `label_ref` | u64 | 字符串池句柄 |
| `confidence` | f32 | 最高后验 |
| `posterior_ref` | u64 | 后验数组句柄（Top-K） |
| `top_k` | u8 | ≤ 16 |

### BRAIN.IN.EXECUTE（Body → Brain）
| 字段 | 类型 | 说明 |
|---|---|---|
| `intent_id` | u64 | 必须命中意图缓冲，否则 `E_NO_INTENT` |
| `max_len` | u16 | 基元链最大长度 |

### BRAIN.HEALTH（Brain → Body）
| 字段 | 类型 | 说明 |
|---|---|---|
| `struct_ok` | u8 | 结构完整性 |
| `weight_overflow_ppm` | u32 | 越界权重占比 |
| `trace_anomaly_ppm` | u32 | 异常迹占比 |
| `degraded` | u8 | `MODULATION_DEGRADED` 等标志 |

### BRAIN.REPAIR（Body → Brain）
| 字段 | 类型 | 说明 |
|---|---|---|
| `checkpoint_id` | u64 | 目标检查点 |
| `trace_policy` | u8 | 1 按 Δt 衰减 2 清零 |

### BRAIN.SLEEP（Body → Brain）
| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | u8 | 1 痕迹槽 >70% 2 不活跃 30min + >50% 3 手动 |
| `replay_budget` | u32 | 最大重放 tick |

### caps / quota（双向）
| 字段 | 类型 | 说明 |
|---|---|---|
| `caps` | u32 | bit0 文本 bit1 图像 bit2 音频 bit3 关节 bit4 记忆 |
| `max_neurons` | u32 | 硬上限 |
| `max_links_per_n` | u16 | 默认 12 |
| `burst_tick_hz` | u16 | 默认 60 |
| `idle_tick_hz` | u16 | 默认 5 |
| `compete_tick_hz` | u16 | 竞争窗口专用，默认 30 |

## 3. 错误码

| 码 | 名称 | 场景 |
|---|---|---|
| 0 | `E_OK` | — |
| 1 | `E_SLEEPING` | 睡眠中收到 EXECUTE |
| 2 | `E_NO_INTENT` | intent_id 不存在或已超时 |
| 3 | `E_QUOTA` | 超神经元/连接配额 |
| 4 | `E_DEGRADED` | 调质降级模式 |
| 5 | `E_FORMAT_TOO_NEW` | 容器版本过高 |
| 6 | `E_RUNNER_TOO_OLD` | runner 版本过低 |
| 7 | `E_BAD_ARG` | 字段非法 |

## 4. 权限矩阵

| 主体 | 读权重 | 写权重 | 写调质 | 触发睡眠 | 结构裁决 |
|---|---|---|---|---|---|
| Body（UI/调度） | 否 | 否 | 是（经端口） | 是 | 否 |
| Body-SNC | 否 | 否 | 否 | 否 | 否 |
| Brain（core） | 是 | 是 | 读 | 执行 | 只读运行 |
| Brain-SNC（forge） | 是 | 是 | 读 | 调度 | 是 |
| neuro-play | 是 | 否 | 否 | 否 | 否 |

## 5. 版本与兼容

- 信封 `schema_version` 不匹配 → 拒绝并记录 `health.jsonl`。
- 追加字段必须给默认值；删除/改义必须升主版本。
- 端口消息不得传裸指针；跨进程只传句柄 + 长度 + 偏移。

## 6. M5 落地清单

1. 写 `spec/link.fbs`（FlatBuffers），与本表逐字段对齐；
2. 生成 C 头与 Python 绑定；
3. 加 schema 兼容性回归（旧消息新 runner、新消息旧 runner）；
4. 用 `neuro-core` 空桩跑通一次 PROPOSE→EXECUTE 往返。

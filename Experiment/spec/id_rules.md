# ID 规则 · 定稿 v1（M0 冻结）

> 依据：设计定稿 §3.1 / §3.2；实现：`core/neuro_types.h`、`core/neuro_id.c`
> 验收：`build_m0.ps1`（71 项检查）

## 1. 字符串形态

```
TYPE-BATCH-PARENT-INDIV
```

| 段 | 长度 | 字符集 | 说明 |
|---|---|---|---|
| TYPE | 2–4 | ASCII | 类型码，见 §4 |
| BATCH | 4 | base36（0-9A-Z） | 批次索引，上限 1,679,615 |
| PARENT | 4 | base36 | 父索引，上限 1,679,615 |
| INDIV | 6 | base62（0-9A-Za-z） | 32 位随机码，零填充 |

**总长 = 19 / 20 / 21（对应 TYPE 长度 2 / 3 / 4）。**

**裁决**：设计稿「ID 总长 18 字符」为笔误。按 2+1+4+1+4+1+6 计算，TYPE=2 时已是 19；18 无法由任何分段组合得到。以本文为准。

例：`SVI-08IX-0IU-00z3fQ`

## 2. 解析规则

- 必须恰好 4 段，以 `-` 分隔；多一段或少一段均拒绝（`NEURO_E_ID_PARSE`）。
- 段长必须严格匹配上表；TYPE 超长/过短直接拒绝。
- 字符集外字符拒绝；INDIV 解码值 > 2³²−1 拒绝（`NEURO_E_ID_RANGE`）。
- 解析失败不写输出缓冲，调用方需自行清零。

## 3. 热路径存储

```c
typedef struct {                 /* 24 B, packed */
    uint64_t id_hash;            /* FNV-1a 64 of canonical string */
    uint32_t type_code;          /* 4 ASCII chars, right padded with 0 */
    uint32_t batch, parent, indiv;
} NeuronKey;
```

- 热路径**只**用 `id_hash + type_code + 三段索引`，禁止格式化字符串。
- 完整可读串放冷数据 `string_pool` 区块，索引 `id_hash → offset`（按哈希排序，二分或开放寻址）。
- 哈希：FNV-1a 64，offset basis `0xcbf29ce484222325`，prime `0x100000001b3`。
- 碰撞：插入时比对字符串池中的完整串；冲突则重试 INDIV（同 master 种子流的下一个值），最多 8 次；仍冲突则报错并记录 `health.jsonl`。

## 4. 类型码

- 4 字节 ASCII 打包：`c0<<24 | c1<<16 | c2<<8 | c3`，3 字符类型末字节为 0。
- 域位（T1）：S 感知、O 本体感、A 动作、N 导航、P 处理、M 记忆、D 决策、E 表达、R 测量、F 安全、G 调制。
- `PDc`、`PPm`、`GIn` 是 3 字符特殊类型，归属 P/G 域；其他域不得使用以 `Dc`/`Pm` 结尾的类型码。
- 端口兼容性由 T1+T2 决定，规则表放 `spec/link_schema.md`，不在 ID 层校验。

## 5. 脑编号与文件名（与神经元 ID 无关）

| 对象 | 格式 | 长度 |
|---|---|---|
| 脑编号 | `BBBB-TTTT-AAAAAAAA-CCCC` | 23 |
| 文件名 | `BBBB-TTTT-CCCC-RRRR.neuro` | 19 |

- `AAAAAAAA` 是 8 位能力向量（La/Vi/Au/Lo/Me/Em/Mo/Rw 各 2 字符），**随学习更新**。
- `CCCC` 是诞生时统计摘要（如 `5A3N`），**不随学习更新**。
- 文件名不含能力向量，只含诞生编号 + 4 位随机码；能力向量、统计摘要、`display_name` 写入容器 `metadata` 区块。
- **裁决**：文件名（19）≠ 脑编号（23），两者不是同一字符串的截断。

## 6. 验收与遗留

已由 `core/test_m0.c` 覆盖：往返一致、长度 19/20/21、类型码、哈希稳定性与大小写敏感、6 类非法输入、缓冲容量保护。

遗留（M5 前补）：
- 10 万 ID 压力测试（碰撞率、字符串池装载）；
- 大小写不敏感模式（若宿主需要）不在 v1 范围；
- `id_hash` 跨平台一致性（FNV 为纯整数运算，天然一致）。

## 7. 变更规则

本文件冻结后，任何字段长度/字符集变更必须升 `format_version`，并提供 `neuro-migrate` 映射表；禁止原地修改。

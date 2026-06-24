# DSA Prefill Context Parallel（Round-Robin-Split）简明流程

> **启动参数**：`--enable-dsa-prefill-context-parallel --attn-cp-size 8 --dsa-prefill-cp-mode round-robin-split`
>
> **适用模型**：DeepSeek V4、V3.2、GLM DSA、Mistral-Large-3 等 DSA 架构模型。
> 本文以 V4（`deepseek_v4.py`）为主路径，V2/GLM 差异见 [附录 A](#附录-av4--vs-v2glm-差异)。
>
> **全程实例**：为便于理解，下文用 **cp_size=4、16 tokens** 的具体数据贯穿每一步。
> 实际部署通常 cp_size=8、序列更长，但 round-robin 规律完全一致。

---

## 全程实例设定

16 个 token 编号 T0~T15，4 个 CP rank（GPU），每 rank 分到 4 个 stride token：

```
T0  T1  T2  T3  T4  T5  T6  T7  T8  T9  T10 T11 T12 T13 T14 T15
```

---

## 1. 参数初始化与自动配置

`server_args.py` 中 `_handle_context_parallelism()` 校验核心公式：

```
attn_tp_size = tp_size // attn_cp_size // attn_dp_size     # distributed/parallel_state.py:1931
moe_tp_size  = tp_size // moe_ep_size  // moe_dp_size      # distributed/parallel_state.py:2003
```

round-robin-split 自动配置（[server_args.py:1870-1882](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1870-L1882)）：

```python
self.enable_dp_attention = True
self.moe_dense_tp_size = 1
self.attn_cp_size = self.tp_size // self.dp_size
# 注：不自动设 moe_a2a_backend 和 ep_size（对比 in-seq-split 自动设 deepep + ep_size=tp_size）
```

**本例**：`tp=4, cp=4, dp=1` → `attn_tp_size = 4//4//1 = 1`（attention 权重不做 TP 切分）。

---

## 2. CP 元数据准备

进入 `DeepseekV4ForCausalLM.forward()` 后，首先检查能否启用 CP：

```python
if self.dsa_enable_prefill_cp:
    if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
        forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(...)
        # round-robin-split 返回空 ContextParallelMetadata()（不需要 zigzag 索引）
```

后续各层通过 `dsa_use_prefill_cp(forward_batch)` 判断是否使用 CP。

---

## 3. Round-Robin Split 分发 Token

所有 rank 先对完整 `input_ids` 做 embedding，**然后** 才 split。

### Stride 切片核心代码

`dsa_cp_round_robin_split_data()`（[dsa/utils.py:98](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L98)）——核心就一行：

```python
return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()
```

### 实例：16 tokens → 4 ranks

```
Step 1: 原始 hidden_states [16, D]

  T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15

Step 2: view(-1, 4, D) → reshape 为 [4, 4, D]

  cp_rank →  0     1     2    3
          ┌───────────────────────┐
  组 0    │  T0 │  T1 │  T2 │  T3 │
  组 1    │  T4 │  T5 │  T6 │  T7 │
  组 2    │  T8 │  T9 │ T10 │ T11 │
  组 3    │ T12 │ T13 │ T14 │ T15 │
          └───────────────────────┘

Step 3: [:, cp_rank] 取列 → 每个 rank 拿到 stride 切片

  rank 0: [T0,  T4,  T8,  T12]     ← 第 0 列，stride=4
  rank 1: [T1,  T5,  T9,  T13]     ← 第 1 列
  rank 2: [T2,  T6,  T10, T14]     ← 第 2 列
  rank 3: [T3,  T7,  T11, T15]     ← 第 3 列
```

每 rank 拿到的 token 编号 = `cp_rank, cp_rank+4, cp_rank+8, ...`（stride = cp_size）。

Position IDs 同理 split。V4 还额外对 `input_ids` 做 split（用于 MoE routing），V2 不做。

---

## 4. 每层 Transformer Block 数据流

### 总览图

```
输入: hidden_states — split 状态（每 rank 4 个 stride token）

  ┌─────────────────────────────────────────────────────────────┐
  │  Self-Attention                                             │
  │                                                             │
  │  Q:  本 rank 的 4 tokens                      shape=[4,h,d]  │
  │  KV: AllGather+Rerange 后完整 16 tokens       shape=[16,...] │
  │                                                             │
  │  → 每 rank 独立计算 4 个 Q 对 16 个 KV 的 attention            │
  │  → 输出仍为 split 状态                       shape=[4, D]     │
  └───────────────────────┬─────────────────────────────────────┘
                          ↓
  ┌────────────────────────────────────────────────────────────┐
  │  MLP / MoE（非 a2a / TP 模式）                               │
  │                                                            │
  │  ① AllGather（无 rerange）  [4,D] × 4 ranks → [16,D]       │
  │  ② MoE per-token 计算       [16,D] → [16,D]（TP partial）  │
  │  ③ ReduceScatter            [16,D] → [4,D]（sum + split）  │
  └───────────────────────┬────────────────────────────────────┘
                          ↓
输出: hidden_states — split 状态（传入下一层）

通信/层: KV AllGather ×1 + MLP AllGather ×1 + ReduceScatter ×1 = 3 次（MoE 内部 AllReduce 被 use_reduce_scatter=True 跳过）
```

### 4.1 Attention：KV AllGather + Rerange

每 rank 只计算自己 4 个 Q 的 KV，然后 AllGather 收集所有 rank 的 KV 并恢复 token 顺序：

```python
kv = cp_all_gather_rerange_output(kv.contiguous(), self.cp_size, forward_batch, stream)
```

**实例：KV 的 AllGather + Rerange 过程**

```
① 各 rank 的 KV（split 状态，每 rank 4 个 token 的 KV）：

  rank 0: KV[T0, T4, T8, T12]
  rank 1: KV[T1, T5, T9, T13]
  rank 2: KV[T2, T6, T10,T14]
  rank 3: KV[T3, T7, T11,T15]

② AllGather 直接拼接（按 rank 顺序，交错排列）：

  KV[T0,T4,T8,T12, T1,T5,T9,T13, T2,T6,T10,T14, T3,T7,T11,T15]
   └── rank0 ──┘  └── rank1 ──┘  └── rank2 ──┘  └── rank3 ──┘

③ view(4, 4, ...) → reshape 为 [4, 4, kv_dim]：

          token_idx→  0    1    2    3
  rank 0 行  │ T0 │ T4 │ T8 │T12 │
  rank 1 行  │ T1 │ T5 │ T9 │T13 │
  rank 2 行  │ T2 │ T6 │T10 │T14 │
  rank 3 行  │ T3 │ T7 │T11 │T15 │

④ transpose(0,1) → 行列互换：

          rank_idx→  0    1    2    3
  token组0  │ T0 │ T1 │ T2 │ T3 │
  token组1  │ T4 │ T5 │ T6 │ T7 │
  token组2  │ T8 │ T9 │T10 │T11 │
  token组3  │T12 │T13 │T14 │T15 │

⑤ reshape(16, ...) → 按行展平：

  KV[T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15]
  ✓ 恢复为原始 token 顺序！
```

每个 rank 现在用自己 4 个 Q 对完整 16 个 KV 做 attention → 输出仍为 split 状态 `[4, D]`。

Attention TP=1（每卡持有全部 heads），无需 TP 通信。

### 4.2 MLP/MoE：AllGather → AllReduce → ReduceScatter

**核心代码**（V4 显式调用，V2 通过 `DSACPLayerCommunicator` 隐式处理）：

```python
# MLP 前：AllGather（无 rerange，MLP 是 per-token 操作，顺序不影响结果）
hidden_states = dsa_cp_gather_hidden_states(hidden_states)    # [4,D] → [16,D]

# MoE forward：use_reduce_scatter=True 跳过内部 AllReduce
hidden_states = self.mlp(hidden_states, ..., use_reduce_scatter=_use_cp)

# MLP 后：ReduceScatter（同时完成 TP reduce + CP scatter）
hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)  # [16,D] → [4,D]
```

**实例：MLP 全流程的 token 矩阵变化**

```
① Attention 输出（split 状态）：

  rank 0: H[T0,  T4,  T8,  T12 ]   shape=[4, D]
  rank 1: H[T1,  T5,  T9,  T13 ]
  rank 2: H[T2,  T6,  T10, T14 ]
  rank 3: H[T3,  T7,  T11, T15 ]

② AllGather（无 rerange）→ 每 rank 拿到完整 16 tokens（交错排列）：

  [H(T0),H(T4),H(T8),H(T12), H(T1),H(T5),H(T9),H(T13), H(T2),H(T6),H(T10),H(T14), H(T3),H(T7),H(T11),H(T15)]
   └───── rank0 块 ─────┘    └───── rank1 块 ─────┘    └───── rank2 块 ─────┘    └───── rank3 块 ─────┘
   交错 ≠ 原始顺序，但 MoE 是 per-token 操作，不影响计算结果

③ MoE per-token 计算（每卡用 1/4 权重算 TP partial result）：

  [P(T0),P(T4),P(T8),P(T12), P(T1),P(T5),P(T9),P(T13), P(T2),P(T6),P(T10),P(T14), P(T3),P(T7),P(T11),P(T15)]
   P = TP partial result（4 卡各算 1/4 权重，结果未求和）

④ ReduceScatter = reduce（求和 partial results）+ scatter（按 CP rank 切分）：

  rank 0: R[T0,  T4,  T8,  T12 ]   R = sum of partial results across 4 ranks
  rank 1: R[T1,  T5,  T9,  T13 ]   每个 rank 得到自己 stride token 的完整 MoE 结果
  rank 2: R[T2,  T6,  T10, T14 ]
  rank 3: R[T3,  T7,  T11, T15 ]

  ✓ 回到 split 状态，传入下一层
```

**为什么 CP 用 AllGather+ReduceScatter 而不是 AllReduce？**

ReduceScatter **同时完成 TP reduce（求和 partial results）和 CP scatter（回到 split 状态）**，将 "MoE 内部 AllReduce + 维持 split 状态" 两次通信融合为一次：

```python
# use_reduce_scatter=True → should_skip_post_experts_all_reduce() 返回 True → 跳过 MoE 内部 AllReduce
# ReduceScatter 一步完成: sum(partial results) + split by CP rank
```

---

## 5. 最后一层 → Logits

最后一层后做一次 AllGather + Rerange 恢复完整序列：

```python
# V4: all-gather → hc_head → norm
hidden_states = cp_all_gather_rerange_output(hidden_states, self.cp_size, forward_batch, stream)
hidden_states = self.hc_head(hidden_states, ...)   # mHC 压缩 [16, hc_mult, D] → [16, D]（V4 特有）
hidden_states = self.norm(hidden_states)           # RMSNorm

# V2: norm → all-gather（顺序不同，数学等价，V2 无 mHC）
hidden_states = self.norm(hidden_states, residual)
hidden_states = cp_all_gather_rerange_output(hidden_states, ...)
```

**实例：最终恢复**

```
① 最后一层输出（split 状态）：

  rank 0: H[T0,  T4,  T8,  T12 ]
  rank 1: H[T1,  T5,  T9,  T13 ]
  rank 2: H[T2,  T6,  T10, T14 ]
  rank 3: H[T3,  T7,  T11, T15 ]

② AllGather + Rerange（同 KV AllGather 的过程）：

  → [H(T0), H(T1), H(T2), ..., H(T15)]   正序，完整 16 tokens

③ V4: flatten → hc_head → norm → logits
   V2: 直接 logits
```

---

## 6. Decode 阶段

Prefill 完成后，`ForwardMode.DECODE` 不满足 `is_context_parallel_extend()` → CP 不生效。每 rank 独立处理 1 个 token 的 attention（使用完整 KV cache）。

---

## 7. 两种 MoE 通信模式对比

**实例对比（cp_size=4）**：

### 非 a2a（默认，TP MoE）

```
每层通信时间线（每 rank 视角）：

  ┌──────────┐  ┌───────────┐  ┌───────────────┐  ┌───────────┐  ┌───────────┐
  │ KV AG    │→ │ Attention │→ │ MLP AllGather │→ │ MoE (TP)  │→ │ MLP RS    │
  │ [4]→[16] │  │ Q[4]×KV   │  │ [4]→[16]      │  │ partial   │  │ [16]→[4]  │
  └──────────┘  └───────────┘  └───────────────┘  └───────────┘  └───────────┘
   同步通信①       无通信          同步通信②           无通信          同步通信③

  3 次同步通信/层（MoE 内部 AllReduce 被 use_reduce_scatter=True 跳过，
  由 MLP RS 同时完成 TP reduce + CP scatter），纯串行，不可 overlap
```

### a2a（DeepEP，EP MoE）

```
每层通信时间线（每 rank 视角）：

  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐
  │ KV AG    │→ │ Attention │→ │ A2A dispatch │→ │ MoE (TP=1)   │→ │ A2A combine  │
  │ [4]→[16] │  │ Q[4]×KV   │  │ 定向分发      │  │ 完整权重      │  │ 定向回收      │
  └──────────┘  └───────────┘  └──────────────┘  └───────────────┘  └──────────────┘
   同步通信①       无通信       pairwise ②          无通信            pairwise ③

  1 次同步 + 2 次 pairwise/层，可 overlap
```

### 参数与通信对比表

| | 非 a2a（默认，TP MoE） | a2a（DeepEP，EP MoE） |
|--|---|---|
| **启动参数** | 不传 `--moe-a2a-backend` / `--ep-size` | `--moe-a2a-backend deepep --ep-size <tp_size>` |
| **moe_tp_size** | `tp / 1 / 1 = tp_size`（TP 切分） | `tp / tp / 1 = 1`（无 TP 切分） |
| **Expert 分布** | 每卡全部 expert 的 1/tp 权重 | 每卡 1/tp 个完整 expert |
| **KV AllGather** | `cp_all_gather_rerange_output` | `cp_all_gather_rerange_output` |
| **MLP 前** | AllGather | A2A dispatch（替代） |
| **MoE 内部** | AllReduce（`use_reduce_scatter` 跳过） | 无 AllReduce（`moe_tp_size=1`） |
| **MLP 后** | ReduceScatter | A2A combine（替代） |
| **每层同步通信** | **3 次** | **1 次** + 2×A2A |

---

## 附录 A：V4 vs V2/GLM 差异

| 维度 | V4（`deepseek_v4.py`） | V2（`deepseek_v2.py`，GLM DSA 等） |
|------|------------------------|--------------------------------------|
| 入口类 | `DeepseekV4ForCausalLM` | `DeepseekV2ForCausalLM`（GLM 继承） |
| hidden_states 维度 | 3D `[seq, hc_mult, D]`（mHC） | 2D `[seq, D]` |
| Embedding 后 | `unsqueeze(1).repeat(1, hc_mult, 1)` | 无 mHC |
| CP 元数据 | 含 `apply_cp_reindex()` + `init_flashmla_related()` | 仅 `prepare_context_parallel_metadata()` |
| input_ids 处理 | `cp_round_robin_input_ids()` + `input_ids_global` | 不处理，不传 |
| Attention TP | `__init__` 显式 `tp_rank=0, tp_size=1` | `attn_tp_size` 由公式隐式为 1 |
| KV AllGather | `_compute_kv_bf16()` → `cp_all_gather_rerange_output(kv)` | `rebuild_cp_kv_cache()` → `cp_all_gather_rerange_output(latent)` |
| MLP 通信 | 显式 `dsa_cp_gather/reduce_scatter` | `DSACPLayerCommunicator` 隐式处理 |
| 最后阶段 | AllGather → `hc_head` → `norm` | `norm` → AllGather（数学等价，顺序不同） |
| Residual | `prev_residual/prev_post/prev_comb`（mHC） | 独立 `residual` tensor |

---

## 附录 B：关键函数索引

| 函数 | 文件 | 作用 |
|------|------|------|
| `_handle_context_parallelism` | [server_args.py:3218](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3218) | CP 参数校验 |
| `can_dsa_cp_split` | [dsa/utils.py:175](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L175) | 判断序列能否做 CP split |
| `can_dsa_prefill_cp_round_robin_split` | [dsa/utils.py:85](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L85) | 判断 forward_batch 能否启用 RR CP |
| `dsa_use_prefill_cp` | [dsa/utils.py:265](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L265) | 判断当前 forward 是否使用 CP |
| `dsa_cp_round_robin_split_data` | [dsa/utils.py:98](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L98) | 对 tensor 做 round-robin stride 切片 |
| `cp_all_gather_rerange_output` | [cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310) | AllGather + rerange 恢复顺序 |
| `dsa_cp_gather_hidden_states` | [communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55) | MLP 前 AllGather（无 rerange） |
| `dsa_cp_reduce_scatter_hidden_states` | [communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67) | MLP 后 ReduceScatter |
| `should_skip_post_experts_all_reduce` | [moe/utils.py:422](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L422) | CP 下跳过 MoE 内部 AllReduce |
| `DSACPLayerCommunicator` | [communicator_dsa_cp.py:79](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L79) | V2/GLM CP 层间通信管理器 |
| `rebuild_cp_kv_cache` | [deepseek_v2.py:1874](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1874) | V2 KV latent cache AllGather |

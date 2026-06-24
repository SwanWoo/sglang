# DSA Prefill Context Parallel（In-Seq-Split）简明流程

> **启动参数**：`--enable-dsa-prefill-context-parallel --attn-cp-size 8 --dsa-prefill-cp-mode in-seq-split`
>
> **适用模型**：DeepSeek V4、V3.2、GLM DSA、Mistral-Large-3 等 DSA 架构模型。
> 本文以 V4（`deepseek_v4.py`）为主路径，V2/GLM 差异见 [附录 A](#附录-av4--vs-v2glm-差异)。
>
> **对比文档**：本文对应 `in-seq-split` 模式；`round-robin-split` 模式见
> [dsa_prefill_cp_round_robin_split_flow_simplified.md](dsa_prefill_cp_round_robin_split_flow_simplified.md)。
> 两模式共享同一套入口与大部分基础设施，差异点在各 Step 中以
> "**▶ vs round-robin-split**" 标注。
>
> **全程实例**：为便于对比，沿用 **cp_size=4、16 tokens (T0~T15)** 的具体数据贯穿每一步。
> in-seq-split 把序列切成 `2 * cp_size = 8` 个 block，每 rank 持有 zigzag 的两端。

---

## 全程实例设定

16 个 token 编号 T0~T15，4 个 CP rank（GPU），`cp_segment_num = 2 * cp_size = 8`，每 block = 16/8 = 2 tokens：

```
block0  block1  block2  block3  block4  block5  block6  block7
 T0 T1   T2 T3   T4 T5   T6 T7   T8 T9  T10 T11 T12 T13 T14 T15
```

每 rank 持有 **block r + block (2*cp_size-1-r)**（zigzag 两端），共 4 tokens：

```
rank 0: block0 + block7 = [T0, T1, T14, T15]
rank 1: block1 + block6 = [T2, T3, T12, T13]
rank 2: block2 + block5 = [T4, T5, T10, T11]
rank 3: block3 + block4 = [T6, T7, T8,  T9 ]
```

排列约定：`[prev = block_r, next = block_{2*cp_size-1-r}]`，前半 prev、后半 next。

---

## 1. 参数初始化与自动配置

`server_args.py` 中 `_handle_context_parallelism()` 校验核心公式：

```
attn_tp_size = tp_size // attn_cp_size // attn_dp_size     # distributed/parallel_state.py:1931
moe_tp_size  = tp_size // moe_ep_size  // moe_dp_size      # distributed/parallel_state.py:2003
```

**in-seq-split 自动配置**（[server_args.py:1861-1869](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1861-L1869)）：

```python
self.enable_dp_attention = True
self.moe_dense_tp_size = 1
self.moe_a2a_backend = "deepep"          # 强制 DeepEP
self.ep_size = self.tp_size              # 强制 ep == tp
# 限制: moe_dense_tp_size == 1, moe_a2a_backend == deepep,
#       ep_size == tp_size, batch_size == 1
```

**▶ vs round-robin-split**：round-robin-split **不**自动设 `moe_a2a_backend` / `ep_size`，
允许 TP MoE、multi-batch、FP8 KV cache；in-seq-split 强制 DeepEP EP + batch_size=1。

**本例**：`tp=4, cp=4, dp=1` → `attn_tp_size = 4//4//1 = 1`，`ep_size = 4`，`moe_tp_size = 1`。

---

## 2. CP 元数据准备（zigzag 索引）

进入 `DeepseekV4ForCausalLM.forward()` 后（[deepseek_v4.py:1730](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1730)），首先检查能否启用 CP：

```python
if self.dsa_enable_prefill_cp:
    if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
        forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
            len(input_ids), self.cp_rank, self.cp_size,
            forward_batch.seq_lens_cpu.tolist(),
            extend_seqs_len=forward_batch.extend_seq_lens_cpu,
        )
        # in-seq-split: 生成完整 zigzag 索引（split_list / zigzag_index /
        #               cp_reverse_index / reverse_split_len / kv_len_prev_next ...）
```

### 2.1 `can_dsa_cp_split` 门槛（[dsa/utils.py:175](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L175)）

```python
if is_dsa_prefill_cp_round_robin_split():
    cur_cp_seq_len = seq_len // cp_size           # round-robin 门槛
    assert seq_len % cp_size == 0
else:                                              # in-seq-split 分支
    cur_cp_seq_len = seq_len // (cp_size * 2)      # 需 seq_len > cp_size*2 才非零
```

**▶ vs round-robin-split**：in-seq-split 要求 `seq_len > 2 * cp_size` 且按 `2*cp_size` 对齐；
round-robin-split 只要求 `seq_len >= cp_size` 且按 `cp_size` 整除。

### 2.2 `prepare_context_parallel_metadata`（[cp_utils.py:491](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L491)）

in-seq-split 走完整 zigzag 路径（round-robin-split 在 [cp_utils.py:503](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L503) 直接返回空 `ContextParallelMetadata()`）。
bs=1、16 tokens、cp_size=4 的实例：

```
cp_segment_num = 8, 每 block = 2 tokens
split_list        = [2,2,2,2,2,2,2,2]                  # 8 个 block 的大小
zigzag_index      = [0, 7]  (rank0)                    # 本 rank 持有的 block 索引: block_r, block_{7-r}
cp_reverse_index  = [0,2,4,6,7,5,3,1]                  # AllGather 后恢复 block 序的排列
reverse_split_len = [2,2, 2,2, 2,2, 2,2]               # AllGather 后各段大小（每 rank [prev,next]）
per_rank_actual_token = [4,4,4,4]                      # 每 rank 4 tokens
kv_len_prev_list  = [2]   (rank0: sum blk[:1] = block0)   # prev 段 q（block_r）能看到的 KV 累计长度
kv_len_next_list  = [16]  (rank0: sum blk[:8] = block0..7) # next 段 q（block_{7-r}）能看到的 KV 累计长度
actual_seq_q_prev = [2]   (rank0: block0 的 q 长度)
actual_seq_q_next = [2]   (rank0: block7 的 q 长度)
```

> **注**：`kv_len_prev/next` 是从序列起点算起的**累计** KV 长度（[cp_utils.py:622-623](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L622-L623)，`sum(blk[:cp_rank+1])` / `sum(blk[:cp_segment_num-cp_rank])`），代表对应段 q 能 attend 到的 KV 范围右端点。nsa 模式为裸累计（不含 radix cache prefix），indexer 在 `_get_topk_ragged_with_cp` 内部再加 prefix offset（[dsa_indexer.py:1062-1066](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1062-L1066)）。

后续各层通过 `dsa_use_prefill_cp(forward_batch)` 判断是否使用 CP。

---

## 3. Token 分发（Zigzag Split）

所有 rank 先对完整 `input_ids` 做 embedding，**然后** 才 split。

### 3.1 split 核心代码（[cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145) `cp_split_and_rebuild_data`）

```python
# in-seq-split 分支
input_list = torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)
result = torch.cat([input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0)
```

即：按 `split_list` 切成 8 个 block，再按 `zigzag_index = [r, 2*cp_size-1-r]` 取出两块 cat。

`cp_split_and_rebuild_position`（[cp_utils.py:167](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L167)）对 position ids 同理（split 按 `dim=-1`）。

### 3.2 实例：16 tokens → 4 ranks

```
Step 1: 原始 hidden_states [16, D]

  T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15

Step 2: 按 split_list=[2,2,2,2,2,2,2,2] 切成 8 个 block

  block0: [T0, T1]    block4: [T8, T9]
  block1: [T2, T3]    block5: [T10,T11]
  block2: [T4, T5]    block6: [T12,T13]
  block3: [T6, T7]    block7: [T14,T15]

Step 3: 按 zigzag_index 取 [block_r, block_{7-r}] cat

  rank 0: [T0, T1, T14, T15]   ← block0 + block7 (prev=block0, next=block7)
  rank 1: [T2, T3, T12, T13]   ← block1 + block6
  rank 2: [T4, T5, T10, T11]   ← block2 + block5
  rank 3: [T6, T7, T8,  T9 ]   ← block3 + block4
```

**▶ vs round-robin-split**：round-robin 用 `view(-1, cp_size)[:, cp_rank]` 做 stride 切片
（rank r 拿 `T_r, T_{r+4}, T_{r+8}, ...`）；in-seq-split 持有 zigzag 两端的两整块。

### 3.3 `input_ids` 处理（`cp_round_robin_input_ids`，[cp_utils.py:191](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L191)）

V4 在两种模式下都调用此函数，但函数内部按 a2a backend 分流：

```python
if get_moe_a2a_backend().is_none():
    input_ids = input_ids.reshape(-1, cp_size).T.flatten()   # round-robin TP MoE 路径
else:
    input_ids = input_ids[cp_rank::cp_size].contiguous()      # in-seq-split 强制 deepep → 走这里
```

in-seq-split 强制 `moe_a2a_backend = deepep` → 走 `input_ids[cp_rank::cp_size]` 分支（stride 切片，
供 MoE routing 用）。V2 路径不调用此函数。

---

## 4. 每层 Transformer Block 数据流

### 总览图

```
输入: hidden_states — zigzag split 状态（每 rank 4 tokens: prev|next）

  ┌─────────────────────────────────────────────────────────────┐
  │  Self-Attention（prev / next 两段）                          │
  │                                                             │
  │  Q_prev: block_r 的 2 tokens    Q_next: block_{7-r} 的 2    │
  │  KV_prev/next: AllGather+rerange 后完整 16 tokens            │
  │                                                             │
  │  → 分两段做 attention（_get_topk_ragged_with_cp ×2）         │
  │  → 输出仍为 zigzag split 状态 [4, D]                         │
  └───────────────────────┬─────────────────────────────────────┘
                          ↓
  ┌────────────────────────────────────────────────────────────┐
  │  MLP / MoE（强制 DeepEP EP）                                 │
  │                                                            │
  │  ① A2A dispatch（定向分发 token 到 expert 所在 rank）        │
  │  ② MoE per-token 计算（moe_tp_size=1，每卡完整 expert）      │
  │  ③ A2A combine（定向回收）                                   │
  └───────────────────────┬────────────────────────────────────┘
                          ↓
输出: hidden_states — zigzag split 状态（传入下一层）

通信/层: KV AllGather ×1 + A2A dispatch ×1 + A2A combine ×1 = 1 次同步 + 2×A2A
```

### 4.1 Attention：KV AllGather + Rerange + prev/next 分段

每 rank 只计算自己 4 个 Q（分 prev 2 + next 2）的 KV，然后 AllGather 收集所有 rank 的 KV
并恢复 token 顺序：

```python
kv = cp_all_gather_rerange_output(kv.contiguous(), self.cp_size, forward_batch, stream)
```

`cp_all_gather_rerange_output`（[cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310)）在 in-seq-split 走 **复杂 rerange 路径**：

```python
# in-seq-split 分支（非 round-robin）
output_tensor = cp_all_gather_reorganized_into_tensor(input_tensor, cp_size, forward_batch, stream)
outputs_list = torch.split(output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0)
output_tensor = torch.cat([outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index], dim=0)
```

**实例：KV 的 AllGather + Rerange 过程**

```
① 各 rank 的 KV（zigzag split 状态，每 rank 4 tokens）：

  rank 0: KV[T0, T1, T14, T15]      (prev=block0  next=block7)
  rank 1: KV[T2, T3, T12, T13]      (prev=block1  next=block6)
  rank 2: KV[T4, T5, T10, T11]      (prev=block2  next=block5)
  rank 3: KV[T6, T7, T8,  T9 ]      (prev=block3  next=block4)

② AllGather 按 rank 顺序拼接（每 rank 内 prev 在前、next 在后）：

  KV[T0,T1,T14,T15, T2,T3,T12,T13, T4,T5,T10,T11, T6,T7,T8,T9]
   └── rank0 ──┘  └── rank1 ──┘    └── rank2 ──┘   └── rank3 ──┘

③ split by reverse_split_len=[2,2,2,2,2,2,2,2] → 8 段（block 交错序）：

  seg0 = block0 = [T0, T1]     seg4 = block2 = [T4, T5]
  seg1 = block7 = [T14,T15]    seg5 = block5 = [T10,T11]
  seg2 = block1 = [T2, T3]     seg6 = block3 = [T6, T7]
  seg3 = block6 = [T12,T13]    seg7 = block4 = [T8, T9]

④ 按 cp_reverse_index=[0,2,4,6,7,5,3,1] 重排段顺序：

  seg[0],seg[2],seg[4],seg[6],seg[7],seg[5],seg[3],seg[1]
  = block0, block1, block2, block3, block4, block5, block6, block7

  KV[T0,T1, T2,T3, T4,T5, T6,T7, T8,T9, T10,T11, T12,T13, T14,T15]
  ✓ 恢复为原始 block 顺序（block0..block7）！
```

#### prev/next 分段 attention 与 indexer

in-seq-split 的 DSA indexer（[dsa_indexer.py:1484-1506](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1484-L1506)）把本 rank 的 Q/KV 按 prev/next 分两段
分别做 topk：

```python
if forward_batch.attn_cp_metadata is not None and is_dsa_prefill_cp_in_seq_split():
    kv_len_prev      = forward_batch.attn_cp_metadata.kv_len_prev_list[0]
    kv_len_next      = forward_batch.attn_cp_metadata.kv_len_next_list[0]
    actual_seq_q_prev = forward_batch.attn_cp_metadata.actual_seq_q_prev_list[0]
    actual_seq_q_next = forward_batch.attn_cp_metadata.actual_seq_q_next_list[0]

    q_fp8_prev, q_fp8_next       = torch.split(q_fp8, (q_fp8.shape[0]+1)//2, dim=0)
    weights_prev, weights_next   = torch.split(weights, ...)
    topk_result_prev = self._get_topk_ragged_with_cp(...)   # prev 段
    topk_result_next = self._get_topk_ragged_with_cp(...)   # next 段
```

**▶ vs round-robin-split**：round-robin-split 不走此分支，正常 prefill indexer 路径，
无 prev/next 拆分；亦无需 `kv_len_prev/next` 元数据。

Attention TP=1（每卡持有全部 heads），无需 TP 通信。

### 4.2 MLP/MoE：强制 DeepEP A2A dispatch/combine

in-seq-split 强制 `moe_a2a_backend = deepep`、`ep_size = tp_size`、`moe_tp_size = 1`，
所以 MLP 走 A2A 路径（**非** AllGather+ReduceScatter）：

```python
# deepseek_v4.py:1399-1428
_use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
if _use_cp:
    # 非 a2a 分支: dsa_cp_gather_hidden_states (AllGather)
    # in-seq-split 强制 deepep → 走 assert 分支，不做 AllGather
    assert get_moe_a2a_backend().is_deepep(), "CP requires DeepEP ..."

hidden_states = self.mlp(hidden_states, forward_batch,
                         input_ids=input_ids, input_ids_global=input_ids_global,
                         use_reduce_scatter=_use_cp)
# DeepEP MoE 内部: dispatch → expert 计算 → combine，moe_tp_size=1 无 AllReduce
```

**实例：MLP 全流程的 token 矩阵变化（DeepEP EP）**

```
① Attention 输出（zigzag split 状态）：

  rank 0: H[T0, T1, T14, T15]   shape=[4, D]
  rank 1: H[T2, T3, T12, T13]
  rank 2: H[T4, T5, T10, T11]
  rank 3: H[T6, T7, T8,  T9 ]

② A2A dispatch：按 routing 把每个 token 发到持有目标 expert 的 rank
   （dispatch 后 token 顺序与 expert 分布有关，不再是原 rank 的 4 tokens）

③ MoE 计算：每卡持有 1/cp_size 个完整 expert，moe_tp_size=1 → 无 TP partial、无 AllReduce

④ A2A combine：把 expert 输出按原 token 归还到原 rank

  rank 0: R[T0, T1, T14, T15]   回到 zigzag split 状态，传入下一层
  rank 1: R[T2, T3, T12, T13]
  rank 2: R[T4, T5, T10, T11]
  rank 3: R[T6, T7, T8,  T9 ]
```

**▶ vs round-robin-split**：round-robin-split 默认 TP MoE（AllGather → ReduceScatter，MoE 内部 AllReduce 被 `use_reduce_scatter=True` 跳过，
3 次同步通信/层），亦可选 deepep。in-seq-split **只能** deepep（1 次同步 KV AllGather + 2×A2A/层，
可 overlap）。`use_reduce_scatter=True` 在 in-seq-split 下因 `moe_tp_size=1` 实际不触发
ReduceScatter（`should_skip_post_experts_all_reduce` 仍返回 True 跳过内部 AllReduce）。

---

## 5. 最后一层 → Logits

最后一层后做一次 AllGather + Rerange 恢复完整序列（[deepseek_v4.py:1615](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1615)）：

```python
# V4: all-gather → hc_head → norm
if self.pp_group.is_last_rank and dsa_use_prefill_cp(forward_batch):
    hidden_states = cp_all_gather_rerange_output(
        hidden_states, self.cp_size, forward_batch, torch.cuda.current_stream())
hidden_states = self.hc_head(hidden_states, ...)   # mHC 压缩 [16, hc_mult, D] → [16, D]（V4 特有）
hidden_states = self.norm(hidden_states)           # RMSNorm

# V2: norm → all-gather（顺序不同，数学等价，V2 无 mHC）
hidden_states = self.norm(hidden_states, residual)
hidden_states = cp_all_gather_rerange_output(hidden_states, ...)
```

AllGather+Rerange 过程同 4.1（in-seq-split 走 `cp_reverse_index` 复杂 rerange 路径）。

**实例：最终恢复**

```
① 最后一层输出（zigzag split 状态）：

  rank 0: H[T0, T1, T14, T15]
  rank 1: H[T2, T3, T12, T13]
  rank 2: H[T4, T5, T10, T11]
  rank 3: H[T6, T7, T8,  T9 ]

② AllGather + Rerange（cp_reverse_index 恢复 block 序）：

  → [H(T0),H(T1), H(T2),H(T3), ..., H(T14),H(T15)]   正序，完整 16 tokens

③ V4: flatten → hc_head → norm → logits
   V2: 直接 logits
```

---

## 6. Decode 阶段

Prefill 完成后，`ForwardMode.DECODE` 不满足 `is_context_parallel_extend()` → CP 不生效。
每 rank 独立处理 1 个 token 的 attention（使用完整 KV cache）。两模式行为一致。

---

## 7. 调度限制：batch_size == 1

[schedule_policy.py:860](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/schedule_policy.py#L860) 对 in-seq-split 强制单 batch：

```python
# TODO support cp with multiple requests
# Enabling context parallelism currently presents precision issues;
# therefore, the prefill-batch setting is temporarily set to 1.
if (self.dsa_prefill_cp_in_seq_split) and len(self.can_run_list) >= 1:
    return AddReqResult.OTHER
```

**▶ vs round-robin-split**：round-robin-split **无此限制**，支持 multi-batch prefill
（`dsa_cp_round_robin_split_q_seqs` 处理多 seq 的 stride 分发与 `bs_idx` 映射，
见 [dsa/utils.py:236](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L236)）。

---

## 8. 两种模式总对比

| 维度 | in-seq-split | round-robin-split |
|--|--|--|
| **启动参数** | `--dsa-prefill-cp-mode in-seq-split` | `--dsa-prefill-cp-mode round-robin-split`（默认） |
| **自动配置** | 强制 `moe_a2a_backend=deepep` + `ep_size=tp_size` | 不设 a2a/ep，允许 TP MoE |
| **batch_size** | **强制 1**（[schedule_policy.py:860](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/schedule_policy.py#L860)） | 支持多 batch |
| **MoE 通信** | DeepEP A2A dispatch/combine（强制） | TP MoE（AllGather+ReduceScatter）或 DeepEP |
| **FP8 KV cache** | 不支持（TODO，[server_args.py:1862](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1862)） | 支持 |
| **Token 分发** | zigzag 两端 block（`block_r` + `block_{2*cp-1-r}`） | stride 切片（`token_idx % cp_size`） |
| **CP 对齐** | `2 * cp_size`（`get_cp_padding_align_size` 触发 ×2） | `cp_size` |
| **seq_len 门槛** | `> 2 * cp_size`，按 `2*cp_size` 对齐 | `>= cp_size`，按 `cp_size` 整除 |
| **CP 元数据** | 完整 zigzag 索引（`split_list`/`zigzag_index`/`cp_reverse_index`/`kv_len_prev_next`） | 空 `ContextParallelMetadata()`，改用 `apply_cp_reindex()` + `init_flashmla_related()` |
| **Attention Q 分段** | prev/next 两段分别 topk（[dsa_indexer.py:1484](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1484)） | 不分段，正常 prefill indexer |
| **AllGather rerange** | `cp_reverse_index` 复杂 rerange | `view(cp_size,-1).transpose(0,1).reshape` 简单 transpose |
| **每层同步通信** | 1 次（KV AllGather）+ 2×A2A | 3 次（TP MoE：KV AG + MLP AG + MLP RS）或 1 次+2×A2A（DeepEP） |
| **input_ids 处理** | `input_ids[cp_rank::cp_size]`（deepep 分支） | `reshape(-1,cp_size).T.flatten()`（TP）或 stride（deepep） |

---

## 附录 A：V4 vs V2/GLM 差异

| 维度 | V4（`deepseek_v4.py`） | V2（`deepseek_v2.py`，GLM DSA 等） |
|------|------------------------|--------------------------------------|
| 入口类 | `DeepseekV4ForCausalLM` | `DeepseekV2ForCausalLM`（GLM 继承） |
| hidden_states 维度 | 3D `[seq, hc_mult, D]`（mHC） | 2D `[seq, D]` |
| Embedding 后 | `unsqueeze(1).repeat(1, hc_mult, 1)` | 无 mHC |
| CP 元数据 | `prepare_context_parallel_metadata()`（in-seq-split 生成 zigzag；round-robin 返回空） + `apply_cp_reindex()` + `init_flashmla_related()` | 仅 `prepare_context_parallel_metadata()` |
| input_ids 处理 | `cp_round_robin_input_ids()` + `input_ids_global` | 不处理，不传 |
| Attention TP | `__init__` 显式 `tp_rank=0, tp_size=1` | `attn_tp_size` 由公式隐式为 1 |
| KV AllGather | `_compute_kv_bf16()` → `cp_all_gather_rerange_output(kv)` | `rebuild_cp_kv_cache()` → `cp_all_gather_rerange_output(latent)` |
| MLP 通信 | 显式 `dsa_cp_gather/reduce_scatter`（TP）或 A2A（deepep） | `DSACPLayerCommunicator` 隐式处理 |
| 最后阶段 | AllGather → `hc_head` → `norm` | `norm` → AllGather（数学等价，顺序不同） |
| Residual | `prev_residual/prev_post/prev_comb`（mHC） | 独立 `residual` tensor |

---

## 附录 B：关键函数索引

| 函数 | 文件 | 作用 |
|------|------|------|
| `_handle_context_parallelism` | [server_args.py:3218](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3218) | CP 参数校验（整除/互斥） |
| `_handle_model_specific_adjustments` | [server_args.py:1792](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1792) (DSA-CP 子块 `:1861-1869`) | in-seq-split 强制 deepep+ep=tp 的自动配置 |
| `is_dsa_prefill_cp_in_seq_split` | [dsa/utils.py:71](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L71) | 判断当前是否 in-seq-split 模式 |
| `is_prefill_cp_in_seq_split` | [cp_utils.py:66](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L66) | 非 DSA 版（普通 MLA CP）同义判断 |
| `get_cp_padding_align_size` | [cp_utils.py:73](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L73) | in-seq-split 返回 `2*cp_size`，round-robin 返回 `cp_size` |
| `can_dsa_cp_split` | [dsa/utils.py:175](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L175) | in-seq-split 门槛 `seq_len // (cp_size*2)` |
| `prepare_context_parallel_metadata` | [cp_utils.py:491](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L491) | 生成 zigzag 索引（in-seq-split）；round-robin 返回空 |
| `cp_split_and_rebuild_data` | [cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145) | 按 zigzag_index 切 block 并 cat（in-seq-split） |
| `cp_split_and_rebuild_position` | [cp_utils.py:167](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L167) | position ids 同理 zigzag split |
| `cp_round_robin_input_ids` | [cp_utils.py:191](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L191) | input_ids stride/transpose（deepep vs TP 分流） |
| `cp_all_gather_reorganized_into_tensor` | [cp_utils.py:215](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L215) | AllGather + pad/remove（in-seq-split rerange 前置） |
| `cp_all_gather_rerange_output` | [cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310) | AllGather + `cp_reverse_index` 恢复 block 序 |
| `cp_all_gather_rerange_kv_cache` | [cp_utils.py:381](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L381) | KV cache 版 rerange |
| `dsa_use_prefill_cp` | [dsa/utils.py:265](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L265) | 判断当前 forward 是否使用 CP |
| `dsa_cp_gather_hidden_states` | [communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55) | MLP 前 AllGather（TP 模式，in-seq-split 不走） |
| `dsa_cp_reduce_scatter_hidden_states` | [communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67) | MLP 后 ReduceScatter（TP 模式） |
| `DSACPLayerCommunicator` | [communicator_dsa_cp.py:79](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L79) | V2/GLM CP 层间通信管理器 |
| `should_skip_post_experts_all_reduce` | [moe/utils.py:422](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L422) | CP 下跳过 MoE 内部 AllReduce |
| `_get_topk_ragged_with_cp` | [dsa_indexer.py:960](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L960) (调用点 `:1507`) | in-seq-split prev/next 分段 topk |
| `rebuild_cp_kv_cache` | [deepseek_v2.py:1874](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1874) | V2 KV latent cache AllGather |

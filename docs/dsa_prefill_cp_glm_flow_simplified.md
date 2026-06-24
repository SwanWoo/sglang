# SGLang DSA Prefill Context Parallel 流程（GLM 5.2 DSA，精简版）

> 本文是 [`dsa_prefill_cp_glm_flow.md`](dsa_prefill_cp_glm_flow.md) 的轻代码简化版，只保留流程逻辑与关键函数名，省略代码片段与行号。详版见原文件。
>
> **适用模型**：GLM 5.2 DSA（`GlmMoeDsaForCausalLM`，继承 `DeepseekV2ForCausalLM`，走 V2 代码路径）。
>
> **两种 split 模式**（启动参数解析阶段与 MoE 后端绑定，不可解耦）：
>
> | 模式 | MoE 后端 | 自动配置 |
> |------|---------|---------|
> | `round-robin-split` | **TP-only**（`moe_a2a_backend=none`, `ep_size=1`） | `enable_dp_attention`, `moe_dense_tp_size=1` |
> | `in-seq-split` | **DeepEP（EP）** | 上述 + 强制 `moe_a2a_backend=deepep`, `ep_size=tp_size` |
>
> 两模式 attention 侧完全相同（`attn_tp_size=1`），差异仅在 MLP/MoE。分歧点以 **▶ round-robin-split** / **▶ in-seq-split** 标注。

---

## Step 0：参数初始化

`_handle_context_parallelism()` 校验 `tp_size % attn_cp_size == 0`、两 CP flag 互斥。`_handle_model_specific_adjustments()` 按模式自动配置：

- **两模式共用**：`enable_dp_attention=True`、`moe_dense_tp_size=1`、`attn_cp_size = tp_size // dp_size`、`disable_piecewise_cuda_graph=True`
- **▶ in-seq-split** 额外强制 `moe_a2a_backend=deepep`、`ep_size=tp_size`，并约束 `batch_size==1`、不支持 FP8 KV cache
- **▶ round-robin-split** 不设 a2a/ep，保留默认 → TP-only MoE；支持 multi-batch

推导（`parallel_state.py`）：

| | round-robin-split | in-seq-split |
|---|---|---|
| `attn_tp_size` | 1 | 1 |
| `moe_tp_size` | 8（expert 权重 TP 切 8 份） | 1（expert 权重不切，EP） |
| 每卡 expert | 全部 routed expert 的 1/8 权重 | `n_routed/ep_size` 个完整 expert |

---

## Step 1-3：请求 → Scheduler 组 Batch

HTTP 请求 → TokenizerManager tokenize → ZMQ → Scheduler `recv_requests()` → `handle_generate_request()` 入 `waiting_queue`。

`_get_new_batch_prefill_raw()` 取请求组 batch：

- **▶ round-robin-split**：支持 multi-batch
- **▶ in-seq-split**：`PrefillAdder.add_one_req()` 强制 `batch_size==1`（第 2 个起被拒，zigzag prev/next 分段 topk 的多 batch 支持有精度问题）

`prepare_for_extend()` 设 `forward_mode = EXTEND`，构建 `input_ids` / `extend_seq_lens`。

---

## Step 4：CP 元数据准备

`DeepseekV2ForCausalLM.forward()` 调 `can_dsa_cp_split()` 判断能否 CP split：

- **▶ round-robin-split**：门槛 `seq_len // cp_size`，要求 `seq_len ≥ cp_size` 且整除
- **▶ in-seq-split**：门槛 `seq_len // (cp_size * 2)`，要求 `seq_len > 2 * cp_size`

通过后调 `prepare_context_parallel_metadata()`：

- **▶ round-robin-split**：返回**空** `ContextParallelMetadata`（不需 zigzag 索引，split 由 stride 切片完成）
- **▶ in-seq-split**：生成完整 zigzag 索引——`split_list`（各 block 大小）、`zigzag_index`（本 rank 持 `[block_r, block_{2*cp-1-r}]`）、`cp_reverse_index`（AllGather 后恢复 block 序）、`reverse_split_len`、`kv_len_prev/next_list`、`actual_seq_q_prev/next_list`

padding 对齐 `get_cp_padding_align_size()`：round-robin 返回 `cp_size`，in-seq 返回 `2*cp_size`。

> V2 路径不调 V4 专属的 `apply_cp_reindex()` / `init_flashmla_related()`。

`dsa_use_prefill_cp()` 后续各层判断：`attn_cp_metadata is not None and dsa_enable_prefill_cp and is_context_parallel_extend()`。

---

## Step 5：CP Token 分发

`DeepseekV2Model.forward()`：embedding（2D `[seq, hidden_dim]`，无 mHC）后，对 `hidden_states` 与 `positions` 做 CP split。

> V2 不调 `cp_round_robin_input_ids()`、不设 `input_ids_global`（V4 专属，V2 MoE routing 依赖 hidden_states 本身）。

`cp_split_and_rebuild_data()` 两分支：

- **▶ round-robin-split**：`dsa_cp_round_robin_split_data()` → `view(-1, cp_size, ...)[:, cp_rank]` stride 切片
- **▶ in-seq-split**：按 `split_list` 切 `2*cp_size` 个 block，按 `zigzag_index` 取 `[block_r, block_{2*cp-1-r}]` cat

### 矩阵示意（64 tokens, cp_size=8，split 后每 rank 8 tokens）

**▶ round-robin-split**（stride 交错）：

```
view(-1, 8, D) → [:, cp_rank]
rank 0: [t0, t8, t16, t24, t32, t40, t48, t56]   ← 第 0 列
rank 7: [t7, t15, t23, t31, t39, t47, t55, t63]   ← 第 7 列
```

**▶ in-seq-split**（zigzag 两端，每 block = 4 tokens）：

```
split 16 段 block0..block15，zigzag_index (rank r) = [r, 15-r]
rank 0: [t0,t1,t2,t3,   t60,t61,t62,t63]   ← block0 + block15 (prev + next)
rank 7: [t28,t29,t30,t31, t32,t33,t34,t35]  ← block7 + block8
每 rank 8 tokens：前 4 = prev（block_r），后 4 = next（block_{15-r}）
```

`cp_split_and_rebuild_position()` 对 position_ids 同理 split。

round-robin-split 的 multi-batch q 长度由 `dsa_cp_round_robin_split_q_seqs_cpu()` 计算（跨 seq 累积余量，前 remainder 个 rank 多分 1）。in-seq-split 强制 batch_size=1，无需此函数。

---

## Step 6：每层 Transformer Block 计算

`DeepseekV2DecoderLayer.forward()` 通过 `DSACPLayerCommunicator` 管理 attention 与 MLP 通信。单层数据流（每 rank 视角，split 后 8 tokens）：

```
输入 [8, D] (split)
  │
  ▼ Self-Attention（DSA MLA）
  │   Q: 本 rank 8 tokens
  │   KV: AllGather+Rerange 后完整 64 tokens
  │   → 每 rank 独立算 8 Q 对 64 KV（attn_tp_size=1，无 TP 通信）
  │   → 输出 [8, D] (split)
  │
  ▼ MLP / MoE
  │   ▶ round-robin: AllGather → MoE(TP partial) → ReduceScatter
  │   ▶ in-seq:     A2A dispatch → MoE(EP) → A2A combine
  │
输出 [8, D] (split，传入下一层)
```

### 6a. Attention：KV AllGather

MLA forward 路径调 `rebuild_cp_kv_cache()`：把 `[k_nope, k_pe]` 拼成 latent_cache → `cp_all_gather_rerange_output()` 跨 rank 收集并恢复顺序 → 拆回 `k_nope` / `k_pe`。

`cp_all_gather_rerange_output()` 两分支：

- **▶ round-robin-split**：NCCL AllGather 后 `view(cp_size,-1,...).transpose(0,1).reshape(...)` 三步矩阵变换恢复正序
- **▶ in-seq-split**：`cp_all_gather_reorganized_into_tensor`（AllGather + pad/截取）→ 按 `reverse_split_len` 切段 → 按 `cp_reverse_index` 重排段恢复正序

### 6b. DSA Indexer

- **▶ round-robin-split**：走正常 `_get_topk_ragged`（内部也调 `cp_all_gather_rerange_output`），不分 prev/next
- **▶ in-seq-split**：本 rank Q 按 prev/next 分两段分别 topk（`_get_topk_ragged_with_cp` ×2），拼回。断言禁用 piecewise CUDA graph

Attention forward 后输出保持 split 状态 `[8, D]`。

### 6c. MLP 通信

MLP 模式由 `_compute_mlp_mode()` 决定，`DSACPLayerCommunicator` 的 `prepare_mlp` / `postprocess_layer` 据此分流：

- **▶ round-robin-split**：`moe_a2a_backend=none` → `ScatterMode.FULL`
  - `prepare_mlp` → `dsa_cp_gather_hidden_states()`：**MLP 前 AllGather** `[8,D]→[64,D]`（无 rerange，per-token 不关心顺序）
  - `postprocess_layer` → `dsa_cp_reduce_scatter_hidden_states()`：**MLP 后 ReduceScatter** `[64,D]→[8,D]`（sum + 按 rank 分割）

- **▶ in-seq-split**：`moe_a2a_backend=deepep` → `ScatterMode.SCATTERED`
  - `prepare_mlp` / `postprocess_layer` 走 `_simple`/`_trivial` 旁路，**不 gather/scatter**——跨 rank token 流动由 DeepEP A2A 处理

### 6d. MoE 计算

`should_use_reduce_scatter=True`（DSA CP 下）使 `should_skip_post_experts_all_reduce()` 返回 True，跳过 MoE 内部 post-experts AllReduce。

- **▶ round-robin-split**（TP MoE，`moe_tp_size=8`）：入口 `forward_normal`。每卡持 routed expert 的 1/8 权重，算 TP partial result。AllReduce 被 `use_reduce_scatter=True` 跳过，由后续 CP ReduceScatter 统一完成 TP reduce + CP scatter：
  ```
  [8,D] → AllGather [64,D] → MoE(TP partial, 无内部 AR) → ReduceScatter [8,D]
  ```
  非 `forward_normal_dual_stream`（后者要求 fusion-off + capture mode，DSA CP 默认 fusion-on + `disable_piecewise_cuda_graph` 不满足）。

- **▶ in-seq-split**（DeepEP EP，`moe_tp_size=1`）：入口 `forward_deepep`。每卡 `n_routed/ep_size` 个完整 expert，无 TP partial，无 AllReduce。DeepEP A2A dispatch/combine 处理跨 rank token 流动：
  ```
  [8,D] → A2A dispatch → MoE(EP, 完整权重) → A2A combine → [8,D]
  ```
  DeepEP 不走 NCCL，用自研 RDMA+NVLink 拓扑感知 kernel（`DeepEPBuffer`）。

> **shared expert fusion（两路径共用）**：默认 `disable_shared_experts_fusion=False` → shared expert 融进 MoE kernel（`num_fused_shared_experts = n_shared_experts`，top_k +1），`_forward_shared_experts` 返回 None。两路径都如此，与 CP 流程正交。

---

## Step 7：最后一层 → Logits

`DeepseekV2Model.forward()` 末尾（V2 顺序：**先 norm 后 allgather**，RMSNorm 是 per-token 故等价）：

1. `self.norm(hidden_states)`（每 rank 独立对自己的 split norm）
2. `cp_all_gather_rerange_output()` 最终 AllGather + rerange 恢复完整 `[64, D]` 正序

回 `DeepseekV2ForCausalLM.forward()` → `logits_processor` → 采样。

```
各 rank [8,D] (split) → AllGather+rerange → [64,D] (正序) → logits → 采样
```

---

## Step 8-9：回传与 Decode

**回传**：采样得 `next_token_ids` → Scheduler 追加 `output_ids` → ZMQ 发 DetokenizerManager → HTTP 返回。

**Decode**：`ForwardMode.DECODE` → `is_context_parallel_extend()` 返回 False → `dsa_use_prefill_cp()` 返回 False → **CP 不生效**。每 rank 独立处理 1 token attention（用完整 KV cache）。两模式行为一致。

---

## 附录 A：每层通信次数对比

| | ▶ round-robin-split（TP） | ▶ in-seq-split（DeepEP） |
|---|---|---|
| KV AllGather | 1 次（`rebuild_cp_kv_cache`） | 1 次（同左） |
| MLP 前 AllGather | 1 次（`dsa_cp_gather_hidden_states`） | 0（SCATTERED 旁路） |
| MoE 内部 AllReduce | 0（`use_reduce_scatter=True` 跳过） | 0（`moe_tp_size=1`，无 TP partial） |
| MLP 后 ReduceScatter | 1 次（`dsa_cp_reduce_scatter_hidden_states`） | 0（旁路） |
| A2A dispatch/combine | 0 | 2 次（DeepEP `buffer.dispatch`/`combine`） |
| **每层集合通信** | **3 次同步**（KV AG + MLP AG + MLP RS） | **1 次同步 + 2 A2A** |

## 附录 B：两模式核心差异速查

| 维度 | round-robin-split | in-seq-split |
|------|-------------------|--------------|
| MoE 后端 | TP-only（`ep_size=1`） | DeepEP（`ep_size=tp_size`） |
| `moe_tp_size` | 8 | 1 |
| batch_size | 多 batch | 强制 1 |
| Token 分发 | stride 交错（`token_idx % cp_size`） | zigzag 两端（`block_r` + `block_{2*cp-1-r}`） |
| CP 对齐 | `cp_size` | `2*cp_size` |
| seq_len 门槛 | `≥ cp_size`，整除 | `> 2*cp_size` |
| CP 元数据 | 空 | 完整 zigzag 索引 |
| Indexer | 正常 ragged topk | prev/next 分段 topk |
| AllGather rerange | `view.transpose.reshape` | `cp_reverse_index` 段重排 |
| MLP 通信 | AllGather + ReduceScatter | A2A dispatch + combine |
| FP8 KV cache | 支持 | 不支持（TODO） |

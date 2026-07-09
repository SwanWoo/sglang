> SGLang DSA Prefill Context Parallel 精简版（GLM 5.2 DSA）
>
> 完整版见 [dsa_prefill_cp_glm_flow.md](dsa_prefill_cp_glm_flow.md)。本文保留核心数据流、两种 split 模式的关键差异，重点展开 Step 6（层循环）。

---

### 两种模式一览

| | round-robin-split | in-seq-split |
|---|---|---|
| 启动参数 | `--dsa-prefill-cp-mode round-robin-split` | `--dsa-prefill-cp-mode in-seq-split` |
| MoE 后端 | TP-only（`ep_size=1`, `moe_a2a_backend=none`）| DeepEP（`ep_size=tp_size`, `moe_a2a_backend=deepep`）|
| `moe_tp_size` | `tp_size // ep_size = 8`（expert 权重 TP 列切）| `tp_size // ep_size = 1`（expert 不切，EP 行切）|
| `attn_tp_size` | 1（两模式相同，每卡完整 attention 权重）| 1 |
| batch size | 支持 multi-batch（`dp_size` 必须 =1）| 强制 bs=1 |
| 序列切分 | stride 交错 | zigzag 头尾配对 |
| padding 对齐 | `cp_size`（[cp_utils.py:73](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L73)）| `2*cp_size`（zigzag 块需偶数倍）|
| 每层通信 | 3 次 NCCL（KV AG + MLP AG + MLP RS）| 1 次 NCCL + 2 次 A2A（KV AG + dispatch + combine）|

两种模式在启动参数解析阶段与 MoE 后端绑定（[server_args.py:1861-1875](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1861-L1875)），不可解耦。运行时通过全局单例 `_global_server_args` 只读判定模式。

---

### 全流程总览（64 tokens, cp_size=8, 每 rank 8 tokens）

```
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 0  CLI → ServerArgs → 模式绑死 MoE 后端 → 写全局单例       │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 1  HTTP → TokenizerManager → tokenize → ZMQ 发 Scheduler   │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 2  Scheduler 收请求 → waiting_queue                        │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 3  组 batch：PrefillAdder.add_one_req 判能否加入            │
 │         ▶ round-robin: 多 req 可入 (multi-batch)                │
 │         ▶ in-seq:     can_run_list>=1 拒第 2 个 (bs=1)          │
 │         prepare_for_extend → forward_mode=EXTEND                │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 4  forward 入口 DeepseekV2ForCausalLM.forward              │
 │         can_dsa_cp_split 判 fork:                               │
 │           ★ if is_dsa_prefill_cp_round_robin_split():           │
 │               ▶ round-robin: cur = seq // cp                    │
 │               ▶ in-seq:     cur = seq // (2*cp)                 │
 │         prepare_context_parallel_metadata:                      │
 │           ▶ round-robin: 返回空 metadata                        │
 │           ▶ in-seq:     生成 zigzag_index / cp_reverse_index    │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 5  DeepseekV2Model.forward                                 │
 │         embedding [64, D]                                       │
 │         cp_split_and_rebuild_data → 每 rank 8 tokens (split 态)  │
 │           ▶ round-robin: stride 切片 view(-1,cp)[:,rank]        │
 │           ▶ in-seq:     zigzag [block_r, block_{2cp-1-r}]       │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓ 层循环 ×N
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 6  ★★ 每层 DeepseekV2DecoderLayer（本文重点，下文展开）     │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓ 最后一层后
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 7  各 rank [8,D] → AllGather+rerange → [64,D] (正序)       │
 │         → norm → logits_processor → lm_head → 采样 → next_token │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 8  结果回传 → DetokenizerManager → HTTP                    │
 └─────────────────────────────────────────────────────────────────┘
                                   ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │ Step 9  decode: ForwardMode.DECODE → CP 不生效                  │
 │         每 rank 独立 1-token attention（完整 KV cache）          │
 │         两模式行为一致                                          │
 │         原因：1 token 无法切分序列，decode 是 memory-bound       │
 │         decode 的并行靠 batching + TP，不靠 CP                  │
 └─────────────────────────────────────────────────────────────────┘
```

---

### Step 3: in-seq 为什么必须 bs=1（根因）

in-seq 把每 rank 的 Q 按 zigzag 分 **prev / next 两段**分别做 sparse topk（[dsa_indexer.py:1501-1526](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1501-L1526)）。这个"分段 topk"的多 batch 聚合**未实现**。三处证据：

1. **metadata 侧数据齐**：`prepare_context_parallel_metadata()` 用 `for s in range(bs)` 为每个 seq 生成 prev/next 段长。多 seq 的段长数据是有的。
2. **indexer 消费侧只取 `[0]`**：硬编码 `kv_len_prev_list[0]`、`actual_seq_q_prev_list[0]`，bs>1 时 seq 1+ 的数据被丢弃，topk 在错误的 KV 范围上算。
3. **Q 按"对半切"分 prev/next，不分 seq**：`torch.split(q_fp8, (q_fp8.shape[0]+1)//2)` 假设前半=prev、后半=next，单 seq 成立；多 seq 各 seq 边界对不上。

**对比 round-robin**：不分 prev/next，整段 Q 走单路径 `_get_topk_ragged`，多 seq 自然聚合到 ragged offset → multi-batch 无碍。

**round-robin multi-batch 的 q 长度分配**：多 seq 时各 rank q 长度不等分。`dsa_cp_round_robin_split_q_seqs_cpu()`（[dsa/utils.py:221](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L221)）按 stride 切每 seq，余量（`seq_len % cp_size`）分配给前 remainder 个 rank（多 1 token）。结果是各 rank q 长度可能差 1。round-robin 还断言 `dp_size == 1`（[server_args.py:1873-1875](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1873-L1875)），DP attention 不支持与 round-robin CP 并用。

---

### Step 5: CP 序列切分

**round-robin**（stride 切片，[dsa/utils.py:98](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L98)）：

```
view(-1, 8, D)[:, rank]

cp_rank →  0     1      2      3      4      5      6      7
组 0   │  t0  │  t1  │  t2  │  t3  │  t4  │  t5  │  t6  │  t7  │
组 1   │  t8  │  t9  │ t10  │ t11  │ t12  │ t13  │ t14  │ t15  │
  ...
组 7   │ t56  │ t57  │ t58  │ t59  │ t60  │ t61  │ t62  │ t63  │

rank 0 取第 0 列: [t0, t8, t16, t24, t32, t40, t48, t56]   ← 离散交错
rank 7 取第 7 列: [t7, t15, t23, t31, t39, t47, t55, t63]
```

**in-seq**（zigzag 头尾配对，[cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145)）：

```
64 tokens 切 16 块（2×cp_size），每块 4 token

rank 0: [t0,t1,t2,t3]       + [t60,t61,t62,t63]  ← block0 + block15 (最头+最尾)
rank 1: [t4,t5,t6,t7]       + [t56,t57,t58,t59]  ← block1 + block14
  ...
rank 7: [t28,t29,t30,t31]   + [t32,t33,t34,t35]  ← block7 + block8 (中间两块)

每 rank 8 tokens：前 4 = prev（block_r），后 4 = next（block_{15-r}）
```

**为什么用 zigzag**：朴素连续切分在因果 mask 下负载不均（尾 rank 的 q attend 所有 KV，头 rank 只 attend 少量）。zigzag 头尾配对让每 rank 既有头部 token（轻）又有尾部 token（重），负载均衡。且 DSA sparse indexer 需连续段做 prev/next 局部 topk。

---

### ★ Step 6: 每层 Transformer Block（核心）

每层 `DeepseekV2DecoderLayer`（类定义 [deepseek_v2.py:1900](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1900)，forward 在 [:2054](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2054)）分 **attention → MLP** 两大阶段。

#### 单层数据流总览

```
输入: hidden_states [8, D]（split 状态，每 rank 持 8 个 token）

  ┌──────────────────────────────────────────────────────────────────┐
  │  Self-Attention（DSA MLA）                                        │
  │  Q: 本 rank 8 tokens                  shape = [8, heads, d]      │
  │  KV: AllGather+Rerange 后完整 64 tokens  shape = [64, ...]       │
  │  → 每 rank 独立算 8 个 Q 对 64 KV 的 attention（attn_tp_size=1） │
  │  → 输出仍 split 状态                  shape = [8, D]             │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │  MLP / MoE                                                       │
  │  ▶ round-robin: AllGather → MoE(TP partial) → ReduceScatter     │
  │  ▶ in-seq:     A2A dispatch → MoE(EP) → A2A combine              │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓
输出: hidden_states [8, D]（split 状态，传入下一层）
```

#### 层循环调用栈

```python
# DeepseekV2DecoderLayer.forward  (deepseek_v2.py:2054)
hidden, residual = layer_communicator.prepare_attn(...)     # L2066 LayerNorm + 通信
hidden = self.self_attn(...)                                 # L2073 ★ attention（含 KV AllGather）
hidden, residual = layer_communicator.prepare_mlp(...)      # L2088 LayerNorm + (round-robin: AllGather)
hidden = self.mlp(...)                                       # L2118 ★ MoE
hidden, residual = layer_communicator.postprocess_layer(...) # L2133 (round-robin: ReduceScatter)
```

外层 N 层循环在 `DeepseekV2Model.forward`（[deepseek_v2.py:2409](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2409)）：GLM-5.1 共 78 层（3 dense + 75 MoE），每层独立权重，每层重新路由。

---

#### 6a. Self-Attention：MLA 内部完整 10 步

CP 强制走 MLA absorb 路径。机制：`_handle_attention_backend`（[attention_backend_handler.py:74](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_common/attention_backend_handler.py#L74)）检测 CP → `_dispatch_mla_subtype` → 选 `forward_absorb_prepare` + `forward_absorb_core`（非 MHA 路径）。原因：absorb 路径在 `rebuild_cp_kv_cache`（④）处插入 KV AllGather，MHA 路径无此插入点。以下标出各步的 TP/CP 通信和 AllGather 插入点：

```
══════════ forward_absorb_prepare  (forward_mla.py:137) ══════════

① q/kv latent 投影                                     [本地, 无通信]
   q, latent_cache = fetch_qkv_latent()
   k_nope = latent_cache[..., :kv_lora_rank]           # 本 rank 8 token 的 KV
   # attn_tp_size=1 → 权重不切, 每卡完整 (ReplicatedLinear)

② RMSNorm (q_a_layernorm / kv_a_layernorm)             [本地]

③ q_b_proj 展开 q + split nope/rope                    [本地, 无通信]
   q_nope, q_pe = split(q_b_proj(q))
   k_pe = latent_cache[..., kv_lora_rank:]
   # attn_tp_size=1 → q_b_proj 每卡完整 128 head, 不切

④ ★ KV AllGather + rerange  ← CP 核心插入点
   rebuild_cp_kv_cache (deepseek_v2.py:1874):
     latent_cache 拼 [k_nope, k_pe] → cp_all_gather_rerange_output
     → 本 rank KV [8,...] 合成全局 KV [64,...]          [★ NCCL all-gather]

⑤ MLA absorb: q_nope 吸收 kv_b_proj 权重               [本地 bmm]
   q_nope_out = bmm(q_nope, w_kc)
   # w_kc 每卡完整 (attn_tp_size=1)

⑥ RoPE                                                 [本地]

══════════ forward_absorb_core  (forward_mla.py:402) ══════════

⑦ ★ attention: 本 rank Q × 全局 KV                     [本地, 无通信]
   attn_output = attn_mqa(q_nope_out, k_nope, k_nope,
                          forward_batch, q_rope, k_rope)
   # 每卡持完整 128 head (attn_tp_size=1), 用本 rank 8 个 Q 对全局 64 KV 算 attention

⑧ attn_output reshape                                  [本地]

⑨ v absorb: attn_output 吸收 v 部分                     [本地 bmm]
   attn_bmm_output = bmm(attn_output, w_vc)
   # w_vc 每卡完整 (attn_tp_size=1)

⑩ o_proj 输出投影                                       [本地, 无通信]
   output = o_proj(attn_bmm_output)
   # o_proj 构造时 tp_size=1 (get_attention_tp_size()) → all-reduce 不触发
```

| 步 | 操作 | CP 下每卡持有什么 | 跨 rank 通信 |
|---|---|---|---|
| ①②③ | q/kv 投影 + norm + split | attn_tp_size=1，每卡持完整 128 head 权重；数据是本 rank 8 token 的 Q/KV | ❌ 无 |
| **④** | **KV AllGather + rerange** | **各 rank 局部 KV [8] → 合并成全局 KV [64]，恢复原始序** | **✅ NCCL all-gather** |
| ⑤⑥ | q absorb + RoPE | 每卡完整 w_kc/w_vc 权重，本地 bmm（absorb 不切 head） | ❌ 无 |
| **⑦** | **q × 全局 KV attention** | 每卡用本 rank 8 个 Q 对全局 64 KV 算 attention（④ 已 gather） | ❌ 无 |
| ⑧⑨ | reshape + v absorb | 本地 bmm | ❌ 无 |
| ⑩ | o_proj 输出投影 | o_proj 构造时 `tp_size=1`（`get_attention_tp_size()` 在 CP 下 =1）→ `RowParallelLinear.forward` 的 `self.tp_size > 1` 为 False → all-reduce **不触发** | ❌ 无 |

**为什么 AllGather 插在 ④**：①②③ 算出的 KV 是本 rank 8 token 的局部 KV，⑦ attention kernel 需 Q 和**全部 KV** 都齐。必须在 ⑦ 之前 gather。插在 q_b_proj（③）之后、q absorb（⑤）之前——latent_cache 还没被破坏，正好 gather。

##### KV AllGather rerange 详解

`cp_all_gather_rerange_output()`（[cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310)）两模式分支：

**▶ round-robin**（[cp_utils.py:341-358](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L341-L358)）—— transpose 三步恢复正序：

```
gather 前各 rank 的 KV（stride 切片分到的）:
  rank0: [t0,t8,t16,t24,t32,t40,t48,t56]
  rank1: [t1,t9,t17,t25,t33,t41,t49,t57]
  ...
  rank7: [t7,t15,t23,t31,t39,t47,t55,t63]

AllGather 按 rank 顺序拼接 → [64, ...]:
  [t0,t8,..,t56, t1,t9,..,t57, ..., t7,..,t63]

view(8, 8, ...) 看成 [8 rank, 8 token/rank]:
     rank0 → t0  t8  t16 t24 t32 t40 t48 t56
     rank1 → t1  t9  t17 t25 t33 t41 t49 t57
     ...
     rank7 → t7  t15 t23 t31 t39 t47 t55 t63

transpose(0,1) → 按列读:
     t0 t1 t2 t3 t4 t5 t6 t7     (第 0 列 = 各 rank 第 0 个 token)
     t8 t9 ...
     ...
     t56 ... t63

reshape(64) → [t0,t1,t2,...,t63]  ✓ 正序（stride 切片的逆操作）
```

**▶ in-seq**（[cp_utils.py:360-378](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L360-L378)）—— `cp_reverse_index` 重排：

```
各 rank [prev|next] → AllGather 拼接 → split 16 段 → cp_reverse_index 重排
= block0, block1, ..., block15 = t0..t63  ✓ 正序
```

##### DSA Indexer topk

| 模式 | 策略 | 代码 | 原因 |
|---|---|---|---|
| round-robin | 整段 Q 走 `_get_topk_ragged`，不分段。`forward_cuda` 中先调 `_get_q_k_bf16`（[dsa_indexer.py:509](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L509)/[:523](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L523)）做 index key 的 `cp_all_gather_rerange_output`，再调 `_get_topk_ragged`。此 AllGather 与 attention ④ 的 KV latent gather 是**两次独立 AllGather**（不同 tensor） | [dsa_indexer.py:733](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L733) | 交错 token 无法分前后 |
| in-seq | Q 按 prev/next 对半分两段，分别 topk | [dsa_indexer.py:1501-1526](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1501-L1526) | zigzag 的两块天然对应 prev + next，可分段限制 KV 范围 |

---

#### 6b. MLP / MoE：两种通信闭环

这是两种模式差异最大的地方。根因：**round-robin 用 TP MoE（moe_tp_size=8），in-seq 用 EP MoE（moe_tp_size=1）**。

##### MLP 通信模式判定（三级链式）

```
启动期（CLI 解析，定死）
  dsa_prefill_cp_mode = "in-seq-split" / "round-robin-split"
        ↓ server_args.py:1861-1875 (0a 派生)
  in-seq      → moe_a2a_backend = "deepep", ep_size = tp_size
  round-robin → moe_a2a_backend 保持 "none", ep_size = 1
        ↓ 写进 _global_server_args 单例

运行期（每层 forward）
  _compute_mlp_mode() (communicator.py:378) 判 moe_a2a_backend → 选 ScatterMode
        ↓
  DSACPLayerCommunicator 按 ScatterMode 选 AllGather vs A2A 旁路
```

注意：`_compute_mlp_mode` **判的是 `moe_a2a_backend`**（间接由 mode 决定），不是直接判 `dsa_prefill_cp_mode`。但因启动期已绑死，效果等价。

| `dsa_prefill_cp_mode` | `moe_a2a_backend` | `ScatterMode` | MLP 前 | MLP 后 | MoE 跨 rank |
|---|---|---|---|---|---|
| round-robin | `none` | `FULL` | AllGather | ReduceScatter | TP（无 A2A）|
| in-seq | `deepep` | `SCATTERED` | 旁路 | 旁路 | DeepEP A2A |

##### round-robin: AllGather → MoE(TP) → ReduceScatter

```
attention 输出 [8, D] (split, 每 rank 持自己那 8 token)
  │
  ↓ dsa_cp_gather_hidden_states (communicator_dsa_cp.py:55): AllGather（纯拼接，无 rerange）
  │   [8,D] → [64,D]    ← MoE 是 per-token 操作，顺序无关
  │
  ↓ MoE forward_normal (deepseek_v2.py:927):
  │   gate 路由 → 8 expert 并行 GEMM（每 rank 持全部 256 expert 的 1/8 权重）
  │   [64,D] → [64,D] (TP partial)
  │   跳过内部 AllReduce（should_skip_post_experts_all_reduce=True, deepseek_v2.py:1029）
  │
  ↓ dsa_cp_reduce_scatter_hidden_states (communicator_dsa_cp.py:67):
  │   tensor_split(cp_size)[cp_rank] + attn_cp_reduce_scatter_tensor
  │   [64,D] → [8,D]
  │   同时完成 TP reduce（8 卡 MoE partial 求和）+ CP scatter（切回 8 token）
  │
  ↓ hidden_states [8,D] 回 split 态 → 下一层
```

**ReduceScatter 的 reduce 对象是 MoE 的 TP partial，不是 attention**：round-robin 下 `attn_tp_size=1`（attention 无 TP partial），但 `moe_tp_size=8`（expert 权重按 intermediate 维 TP 切，每卡算 partial）。MoE 跳过内部 AllReduce 后，TP partial 推迟到 ReduceScatter 统一 reduce。

**为什么 ReduceScatter 能同时完成 TP+CP**：8 张卡**既是 TP rank 又是 CP rank**（`attn_cp_size == tp_size` 时 `_ATTN_CP = _TP`，CP 复用完整 TP 通信组），ReduceScatter 原语 = Reduce + Scatter，跨这 8 rank 一次完成。

**为什么 MLP 的 AllGather 不做 rerange**：

| | attention KV AllGather | MLP AllGather |
|---|---|---|
| 跨 token 依赖 | ✅ 有（因果 mask）| ❌ 无（per-token routing）|
| AllGather 后 rerange | ✅ 需要（KV 须原始序）| ❌ 不需要（顺序无关）|
| 切回方式 | `cp_reverse_index` 逆重排 | `tensor_split` 纯切（= AllGather 的逆）|

**MLP 通信代码点位**：

| 操作 | 函数 | 代码 |
|---|---|---|
| MLP 前 AllGather | `dsa_cp_gather_hidden_states` | [communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55) |
| 调用点 | `_gather_hidden_states_and_residual` | [communicator_dsa_cp.py:171](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L171) |
| MoE 跳过 AllReduce | `should_skip_post_experts_all_reduce` | [deepseek_v2.py:1029](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1029) |
| MLP 后 ReduceScatter | `dsa_cp_reduce_scatter_hidden_states` | [communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67) |
| └ Scatter 切序列 | `tensor_split(cp_size)[cp_rank]` | [communicator_dsa_cp.py:74](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L74) |
| └ Reduce TP partial | `attn_cp_reduce_scatter_tensor` | [communicator_dsa_cp.py:75](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L75) |
| 调用点 | `_scatter_hidden_states` | [communicator_dsa_cp.py:220](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L220) |
| 通信底层 | `get_attention_cp_group().reduce_scatter_tensor / all_gather_into_tensor` | [dp_attention.py:587](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L587) / [:599](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L599) |

##### in-seq: A2A dispatch → MoE(EP) → A2A combine

```
attention 输出 [8, D] (split)
  │
  │ 无 AllGather（prepare_mlp 走 _simple 旁路，无 gather）
  │
  ↓ DeepEP dispatch (deepep.py): 按 routing 把 token 发给持有其 expert 的 rank
  │   自研 RDMA+NVLink 拓扑感知 kernel，非 NCCL
  │
  ↓ MoE forward_deepep (deepseek_v2.py:1097):
  │   每 rank 32 完整 expert（ep_size=8, 256/8=32），moe_tp_size=1，无 TP partial
  │
  ↓ DeepEP combine: 结果发回原 rank
  │
  │ 无 ReduceScatter（postprocess_layer 走 _trivial 旁路）
  │
  ↓ hidden_states [8,D] → 下一层
```

##### TP 列切 vs EP 行切的几何本质

```
TP 列切 intermediate（每 rank 一列 = 全 256 expert 的 1/8 intermediate）：
                 inter slice0  slice1 ... slice7
                 (256维) (256维)     (256维)
expert 0   ─── [ rank0 │ rank1 │ ... │ rank7 ]
expert 1   ─── [ rank0 │ rank1 │ ... │ rank7 ]
  ...
expert 255 ─── [ rank0 │ rank1 │ ... │ rank7 ]
→ 每 token 选的 8 expert 在 8 卡都有 → 8 卡都得算 → 全广播输入（AllGather）

EP 行切 expert（每 rank 持 32 完整 expert，intermediate 不切）：
                 full inter (2048维, 不切)
expert 0..31  ─── [ rank 0 ]   (32 完整 expert)
expert 32..63 ─── [ rank 1 ]
  ...
expert 224..255 ── [ rank 7 ]
→ 每 token 选的 8 expert 只在持有者 rank → 稀疏路由（dispatch）
```

列切 = 每 token 全卡冗余 = 全广播输入；行切 = 每 token 单卡无冗余 = 稀疏路由。TP 想省 AllGather 只能改 EP。

##### MoE 通信模型对比

| | round-robin (TP) | in-seq (EP) |
|---|---|---|
| MoE 权重切法 | 列切 intermediate | 行切 expert |
| expert 权重 per rank | 256 expert × 1/8 inter | 32 expert × 完整 inter |
| gate 输入 | AG 后 [64, 6144]（全 rank 相同）| 本地 [8, 6144]（各 rank 不同）|
| token 到每卡 | AllGather 全广播 → 8 卡都算同 token | dispatch 路由到 expert 所在卡 |
| 计算冗余 | 8×（8 卡重复算同 token 的 partial）| 1×（每 token 只在 1 卡算）|
| MoE 内 AllReduce | 跳过（推迟至 RS）| 无（moe_tp_size=1）|
| 跨 rank 通信 | AG [64,6144] + RS [64,6144]（NCCL）| dispatch + combine（DeepEP A2A，约 66% 序列量）|

---

#### 6c. MoE 单层内部 5 步

不论 TP 还是 EP，一个 token 进 MoE 到出 MoE 的计算步骤：

```
Step 1  gate（本地，无通信）
  X [n, 6144] @ W_gate [6144, 256] → router_logits [n, 256]
  gate 永远不切（路由需看全 256 expert 才能选 top8）

Step 2  topk（本地，无通信）
  topk_softmax(router_logits, k=8) → topk_ids [n, 8], topk_weights [n, 8]
  每 token 选 8/256 expert

Step 3  dispatch（仅 EP；TP 无此步）
  EP: 按 topk_ids 把 token 发给持有其 expert 的 rank (A2A)
  TP: 无 dispatch，AG 已让全 token 到每卡

Step 4  expert GEMM（8 expert 并行，不是循环 8 轮）
  w13 (gate_up 融合): X → silu(X @ w1) * (X @ w3)   # gate+up GEMM + SiLU 激活
  w2 (down):          → down GEMM → [n, 8, D] 每 expert 一份
  → 8 expert 加权求和 → [n, D]
  I_per_part: TP=256（2048/8），EP=2048（不切）

Step 5  combine / AllReduce（求和）
  TP: 8 expert 的 partial 跨 8 rank ReduceScatter 求和
  EP: combine A2A 发回原 rank 加权求和，无 AR
```

**8 expert 并行**：`w13_weight[topk_ids]` 一次 gather 出 8 个 expert 权重，grouped GEMM kernel 一次算 8 组。

**shared expert**（默认 fusion-on）：shared expert 融进 MoE kernel 作为第 257 个 expert（`top_k = 8+1 = 9`，[deepseek_v2.py:547-551](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L547-L551)）。两模式差异：

| | TP（round-robin）| EP（in-seq DeepEP）|
|---|---|---|
| shared expert 权重 | 列切 inter（每 rank 1/8 维，partial）| 每 rank 一个 shared slot，home rank 持完整 |
| slot 数 | 257（256 routed + 1 shared）| `256 + ep_size`（如 EP=8 → 264，[layer.py:205](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L205)）|
| scaling | 无（全 rank 算 partial，AR 自然求和）| DeepEP fusion 不需 scaling；非 DeepEP 的 EP 用 `1/ep_size`（[deepseek_v2.py:601](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L601)，条件 `not _is_deepep_fusion`）|

---

#### 6d. Dense 层（前 3 层）vs MoE 层

GLM-5.1 共 78 层，**前 3 层 dense**（`first_k_dense_replace=3`，`moe_layer_freq=1`），后 75 层 MoE。dense 层用 `DeepseekV2MLP`（[deepseek_v2.py:211](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L211)），建层时 `_is_layer_sparse`（[deepseek_v2.py:2047](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2047)）分派。

| | Dense MLP（前 3 层）| MoE（后 75 层）|
|---|---|---|
| 是否路由 | 全 token 过同一 MLP | gate 选 8/256 expert |
| 权重切法 | TP 列切 inter（`MergedColumnParallel` + `RowParallel`）| TP 列切 inter / EP 行切 expert |
| CP 通信 | 同 MoE：round-robin AG + RS | 同 |
| 内部 AllReduce | `down_proj` 内置 all_reduce，CP 下同样跳过推迟到 RS | 跳过推迟到 RS / EP 无 AR |

`_compute_mlp_mode`（[communicator.py:378](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator.py#L378)）也先判 `is_layer_sparse`，dense 层走 FULL（与 round-robin MoE 层通信模式一致）。

---

#### 6e. Residual 流向

```
进层: residual [8, D]
  ↓ prepare_attn: input_layernorm(hidden, residual) → hidden_normed, residual
  ↓ self_attn(hidden_normed) → attn_out [8, D]
  ↓ prepare_mlp: post_attention_layernorm(attn_out, residual) → hidden2, residual'
      ↑ residual' = residual + attn_out  (残差累加点 1)
  ↓ mlp(hidden2) → mlp_out
  ↓ postprocess_layer: residual'' = residual' + mlp_out  (残差累加点 2)
出层: hidden_states = residual''  → 下一层
```

**CP 下 residual 同步**：`_gather_hidden_states_and_residual`（[communicator_dsa_cp.py:171](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L171)）**只 gather `hidden_states`，residual 不 gather**——两模式下 residual 都保持 SCATTERED（本地）。因为残差是 per-token 加法，与 CP 切分正交，不需跨 rank 同步。round-robin 的 gather 只作用于 MLP 输入（hidden），ReduceScatter 后 hidden 回 split 态再和本地 residual 相加。

---

#### 6f. TBO 分支（MoE 层双 batch 重叠）

`DeepseekV2Model.forward`（[deepseek_v2.py:2397-2449](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2397-L2449)）：`can_run_tbo` 时 dense 层走 `for` 循环，MoE 层从循环让出交 `model_forward_maybe_tbo`（[L2438](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2438)）双 batch 重叠执行。不开 TBO 时全部层走 `for` 循环。CP 设 `disable_piecewise_cuda_graph=True` 可能影响 TBO 可用性。

---

### 每层通信次数总结

| | round-robin (TP) | in-seq (DeepEP) |
|---|---|---|
| KV AllGather（attention ④）| 1 次（`rebuild_cp_kv_cache` → `cp_all_gather_rerange_output`）| 1 次（同）|
| MLP 前 AllGather | 1 次（`dsa_cp_gather_hidden_states`）| 0（SCATTERED 旁路）|
| MoE 内部 AllReduce | 0（`use_reduce_scatter=True` 跳过）| 0（`moe_tp_size=1`）|
| MLP 后 ReduceScatter | 1 次（`dsa_cp_reduce_scatter_hidden_states`）| 0（旁路）|
| A2A dispatch/combine | 0 | 2 次（DeepEP）|
| **每层合计** | **3 次 NCCL** | **1 次 NCCL + 2 次 A2A** |

通信量：round-robin AG+RS 通信完整 `[64, 6144]`（NCCL 全 rank）；in-seq A2A 通信 sparse 路由后的 token（约 66%），DeepEP 用自研 RDMA+NVLink kernel 拓扑感知。

---

### 关键函数速查

| 函数 | 位置 | 作用 |
|---|---|---|
| `is_dsa_prefill_cp_round_robin_split` | [dsa/utils.py:78](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L78) | 判断 round-robin 模式 |
| `is_dsa_prefill_cp_in_seq_split` | [dsa/utils.py:71](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L71) | 判断 in-seq 模式 |
| `can_dsa_cp_split` | [dsa/utils.py:175](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L175) | forward 门槛：round-robin `seq//cp`, in-seq `seq//(2*cp)` |
| `dsa_use_prefill_cp` | [dsa/utils.py:265](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L265) | 当前 forward 是否用 CP |
| `cp_split_and_rebuild_data` | [cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145) | embedding 后 CP split（stride 或 zigzag）|
| `prepare_context_parallel_metadata` | [cp_utils.py:491](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L491) | round-robin 返空；in-seq 生成 zigzag 索引 |
| `rebuild_cp_kv_cache` | [deepseek_v2.py:1874](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1874) | KV AllGather（两模式共用，attention ④）|
| `cp_all_gather_rerange_output` | [cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310) | AllGather + 恢复正序（transpose 或 cp_reverse_index）|
| `dsa_cp_gather_hidden_states` | [communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55) | MLP 前 AllGather（round-robin only）|
| `dsa_cp_reduce_scatter_hidden_states` | [communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67) | MLP 后 ReduceScatter（round-robin only）|
| `DSACPLayerCommunicator` | [communicator_dsa_cp.py:79](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L79) | 层间通信管理器（prepare_mlp / postprocess_layer）|
| `_compute_mlp_mode` | [communicator.py:378](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator.py#L378) | 判 FULL（round-robin）vs SCATTERED（in-seq）|
| `should_skip_post_experts_all_reduce` | [moe/utils.py:422](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L422) | CP 下跳过 MoE 内部 AllReduce |
| `forward_absorb_prepare` | [forward_mla.py:137](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L137) | MLA absorb prepare（①~⑥，含 KV AllGather ④）|
| `forward_absorb_core` | [forward_mla.py:402](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L402) | MLA absorb core（⑦~⑩，attention + v absorb + o_proj）|
| `cp_all_gather_into_tensor_async` | [parallel_state.py:889](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L889) | CP AllGather 通信底层（pynccl 优先，torch.distributed 兜底）|
| `DeepseekV2MoE.forward_normal` | [deepseek_v2.py:927](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L927) | TP MoE（round-robin 路径）|
| `DeepseekV2MoE.forward_deepep` | [deepseek_v2.py:1097](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1097) | DeepEP EP MoE（in-seq 路径）|
| `dsa_cp_round_robin_split_q_seqs_cpu` | [dsa/utils.py:221](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L221) | round-robin 每 rank q 长度（multi-batch 余量分配）|
| `get_cp_padding_align_size` | [cp_utils.py:73](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L73) | round-robin 返回 `cp_size`，in-seq 返回 `2*cp_size` |
| `_is_layer_sparse` | [deepseek_v2.py:2047](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2047) | 判 dense vs MoE 层（`layer_id >= first_k_dense_replace`）|
| `_handle_attention_backend` | [attention_backend_handler.py:74](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_common/attention_backend_handler.py#L74) | CP 检测 → 强制 MLA absorb 路径 |
| `model_forward_maybe_tbo` | [deepseek_v2.py:2438](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2438) | TBO MoE 层双 batch 重叠 |

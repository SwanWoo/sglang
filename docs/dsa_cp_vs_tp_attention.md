# 为什么 DSA Prefill CP 比 TP 在 attention 上快 7.6×

> 本文档记录一次围绕 profile 数据的多轮代码追问，核心问题：**DSA prefill attention 上，CP（Context Parallel）比 TP（Tensor Parallel）到底优在哪里？为何能差到 7 倍？**
>
> 结论先行：DSA sparse attention 的瓶颈是 **memory-bound 的 indexer scoring（MQA）+ KV gather（共享 latent）**，二者都 ∝ `q_len`、与 128 个 attention head 无关。TP 切 128 head 只动到**非瓶颈的算术部分**；CP 切 `q_len` 直接砍**瓶颈**。MLA 的共享 latent + MQA indexer 让 TP 在 attention 上几乎"没切"，正好被 CP 的 seq 切分治掉。
>
> 本文原理基于以下代码（commit `d8487bad06`）：
> - DSA indexer scoring：[`python/sglang/srt/layers/attention/dsa/dsa_indexer.py`](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py)
> - DSA sparse attention backend：[`python/sglang/srt/layers/attention/dsa_backend.py`](../python/sglang/srt/layers/attention/dsa_backend.py)
> - MLA attention 层：[`python/sglang/srt/models/deepseek_v2.py`](../python/sglang/srt/models/deepseek_v2.py)
> - MLA KV cache 存储：[`python/sglang/srt/mem_cache/memory_pool.py`](../python/sglang/srt/mem_cache/memory_pool.py)
> - `concat_mla_absorb_q` JIT kernel：[`python/sglang/jit_kernel/concat_mla.py`](../python/sglang/jit_kernel/concat_mla.py) / [`csrc/elementwise/concat_mla.cuh`](../python/sglang/jit_kernel/csrc/elementwise/concat_mla.cuh)
> - 并行参数推导：[`python/sglang/srt/distributed/parallel_state.py`](../python/sglang/srt/distributed/parallel_state.py)
>
> 相关文档：
> - DSA Prefill CP 完整流程：[`dsa_prefill_cp_glm_flow.md`](dsa_prefill_cp_glm_flow.md)
> - CP split 可视化：[`dsa_prefill_cp_split_visualization.ipynb`](dsa_prefill_cp_split_visualization.ipynb)
> - RR vs DeepEP MoE 性能：[`deepep_ep_vs_tp_perf.md`](deepep_ep_vs_tp_perf.md)

---

## 0. 测试设定与 profile 数据

测试场景：长 prefill（约 10k 输入 + 1 输出），64 个 sonnet prompt 各 pad 到 ~10k token，单机 8 GPU。

| 配置 | 每卡处理 q_len | sparse_attn 每卡时间 | 加速比 |
|---|---|---|---|
| no-CP TP=8 | ~93k（全序列） | 250.4s / 8 = 31.3s | baseline |
| CP=8 | ~11.6k（1/8 序列） | 33.0s / 8 = 4.1s | **7.6×** |

核心矛盾：CP 只切了 seq（Q 的序列维），TP 只切了 head（Q 的 128 个 head），**为啥只切 seq 比 只切 head 快 7.6 倍**？

---

## 1. DSA sparse attention 在干啥：四步

DSA（DeepSeek Sparse Attention）不是普通 dense attention。普通 attention：每个新 token 看全部旧 token（全 seq 93k 个 KV），算分加权和。DSA：先**挑出最相关的 512 个旧 token**，只看这 512 个 → 省算力。

挑 + 看分四步：

```
Step 1  打分(indexer scoring)  每个新 token 给所有旧 token 打分
Step 2  选 topk                每行挑分数最高的 512 个,扔掉其余 92.5k 个
Step 3  捞 KV (sparse gather)  按 512 个编号从 KV cache 取出 KV,准备做 attention
Step 4  算 attention           新 token 只对这 512 个挑中的 KV 算加权和 → 输出
```

**前三步 = "挑"**，第四步 = "看"。普通 attention 没前三步，直接看全 93k。DSA 加了"挑"这步，只看 512 → 省算力，但"挑"本身要读全 KV 打分 + 捞 KV，变成 **memory 瓶颈**。

### 1.1 Step 1 — 打分（indexer scoring，MQA）

每个新 token 给所有 93k 旧 token 打个相关性分数。

代码 [`dsa_indexer.py:837`](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L837)：

```python
logits = deep_gemm.fp8_mqa_logits(q_fp8[:q_offset], kv_fp8, weights[:q_offset], ks, ke)
#                             q_fp8   [93k, index_n_heads(64), index_head_dim(128)]  新 token 查询
#                             kv_fp8  [93k, 128]                                       旧 token key(单共享头)
#                             → logits [93k, 93k]   分数表
assert logits.shape[0] == len(seq_lens_expanded)   # q_len
assert logits.shape[1] == k_offset                  # 全 seq KV
```

输出 `[q_len, seq_len]` 分数表：第 `i` 行第 `j` 列 = 新 token `i` 对旧 token `j` 的相关性分数。总元素数 `93k × 93k ≈ 86 亿`。

**MQA 意味着**：评分用 1 个共享 K 头（`wk` 输出单头，[`dsa_indexer.py:354`](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L354)），不是 128 个 attention head 各打一份。所以这步**跟那 128 个 attention head 无关** → TP 切 128 head 碰不到这步。

**memory-bound**：算分要读全 93k 个旧 token 的 K（从 KV cache 读），读量巨大，GPU 带宽吃满，算力闲着。

### 1.2 Step 2 — 选 topk（挑 512 个）

每行挑分数最高的 512 个旧 token，扔掉其余 92.5k 个。

代码 [`dsa_indexer.py:1119`](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1119)：

```python
topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1]
# logits [93k, 93k] → topk_indices [93k, index_topk(512)]
```

输出索引表 `[93k, 512]`：第 `i` 行 = 新 token `i` 挑中的 512 个旧 token 的编号。`512` 固定（`index_topk`），不随序列长变。

### 1.3 Step 3 — 捞 KV（sparse KV gather）

按挑出来的编号，从 KV 仓库把那 512 个旧 token 的 value 取出来，准备做加权和。

代码 [`dsa_backend.py:1779`](../python/sglang/srt/layers/attention/dsa/dsa_backend.py#L1779)：

```python
indices_input = page_table_1.unsqueeze(1)   # (s_q, 1, topk) 每 Q token 的 512 个页号
o,_,_ = flash_mla_sparse_fwd(
    q=q_input,            # (s_q, 128_heads, 576)
    kv=kv_cache,          # 完整 KV cache（全 seq, paged）
    indices=indices_input # 每 Q token 只取这 512 页 ← 这就是 gather
)
```

这步 = **按索引取数**：不是读全 93k，只读每行挑中的 512 页。

- 读量 = `q_len × 512 × 576`
- `576` = MLA 压缩 latent（`kv_lora_rank 512 + qk_rope 64`），全 128 head 共享一份 → 跟 head 数无关

**memory-bound**：纯粹读 KV 内存，不怎么做算术。GPU 带宽吃满。

### 1.4 Step 4 — 算 attention（compute-bound）

新 token 只对挑中的 512 个旧 KV 算加权和。这时 128 个 attention head 才全部上场，每 head 对 512 个 KV 算 attention。这步是**算**为主（compute-bound），跟 head 数有关，**TP 能切**。

---

## 2. 三块成本 × 两种并行

| 成本 | 公式 | 瓶颈类型 | TP=8 切 head | CP=8 切 q_len |
|---|---|---|---|---|
| ① indexer scoring | `q_len × seq_len` | **memory+compute** | ❌ MQA 单共享头，与 128 head 无关 | ✅ /8 |
| ② KV gather（取 topk 页） | `q_len × topk × 576` | **memory-bound** | ❌ 576 共享 latent，全 head 读同一份 | ✅ /8 |
| ③ attention 算术 | `q_len × num_heads × topk × v_dim` | compute-bound | ✅ head/8 | ✅ q_len/8 |

> 注：① = Step 1+2（打分 + topk），② = Step 3（捞 KV），③ = Step 4（算 attention）。

**谁是瓶颈**：长 seq prefill 时，①② 是 **memory-bound**（读 KV、评分），∝ `q_len`，与 attention head 数无关。③ 是 compute-bound，∝ `q_len × 128_heads`，TP 切 head 能 8×。**但瓶颈是 ①②，不是 ③。**

---

## 3. 为什么 TP 切不到 ①②

### 3.1 MLA 把 KV 压成共享 latent → ② 切不动

MLA 不存展开的 K/V，存压缩 latent。存储类 `MLATokenToKVPool`，[`memory_pool.py:1805`](../python/sglang/srt/mem_cache/memory_pool.py#L1805)：

```python
self.kv_buffer = [
    torch.zeros((self.size + page_size, 1, self.kv_cache_dim), dtype=store_dtype, device=device)
    for _ in range(self.layer_num)
]
# kv_cache_dim = kv_lora_rank(512) + qk_rope_head_dim(64) = 576
```

- 形状 `[size+page, 1, 576]`，**dim 1 = `1`**（MLA 是 MQA，所有 128 head 共享一个 KV）
- 576 维拆分：前 512 = `k_nope`（压缩 latent），后 64 = `k_pe`（rope）
- 压缩比：普通 MHA 每 token 存 `128×192 + 128×128 = 40960` 维（每 head 一份 K+V），MLA 只存 576 维 → **71× 压缩**

TP 切 128 Q head → 每卡 16 head（[`deepseek_v2.py:1480`](../python/sglang/srt/models/deepseek_v2.py#L1480) `num_local_heads = num_heads // attn_tp_size`），但 KV cache 还是那份 576 → **每卡复制全 KV cache**（无切分）。128 head 读同一份 576 → 每卡读同样的 KV → ② 无并行收益。

对比普通 MHA（TP 有用的情况）：KV = `128 × 192`，TP 切 head → KV 也切 8 份 → 每卡只读 1/8 KV。**MLA 没这待遇**——KV 太省了，没东西给 TP 切。

### 3.2 indexer 是独立小网络（MQA）→ ① 切不动

DSA indexer 是独立的小评分网络，不是 128 个 attention head：

代码 [`dsa_indexer.py:346-366`](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L346)：

```python
self.wq_b = ReplicatedLinear(q_lora_rank, self.n_heads * self.head_dim, ...)   # n_heads = index_n_heads(64)
self.wk   = ReplicatedLinear(hidden_size, self.head_dim, ...)                 # 单头!
self.weights_proj = ReplicatedLinear(hidden_size, self.n_heads, ...)           # 评分头
```

- indexer 的 `n_heads` = `index_n_heads`（DSv4 默认 64），**不是** attention 的 128 head
- `wk` 输出单头（`head_dim`），`ReplicatedLinear` → **每卡完整，不随 TP 切**
- scoring 调用 `fp8_mqa_logits`（**M**QA，多 Q 头共享 1 KV 头）→ 输出 `[q_len, seq_len]`，整个 scoring 与 128 attention head 无关

TP 切 128 head 碰不到这步 → 八卡各读全 KV、各算同样的 `93k×93k` 评分表。

### 3.3 小结：TP 只切到 ③（非瓶颈）

| 步 | TP=8 每卡干啥 | 切到没 |
|---|---|---|
| ① scoring + topk | 读全 KV + 算 `[93k,93k]` 评分 | ❌ 每卡同样 |
| ② KV gather | 读 `93k×512×576` | ❌ 每卡读全 |
| ③ attention 算术 | 16 heads 加权和 | ✅ /8 头 |
| ④ o_proj + AllReduce | RowParallel + AllReduce | ✅ 切 + 通信 |

TP 只在 ③④（`q_b_proj`/`kv_b_proj` ColumnParallel → 16 heads，`o_proj` RowParallel + AllReduce）生效。①② 八卡全冗余重复。

---

## 4. no-CP TP=8 每卡执行流程

设：8 卡，每卡持 128/8 = 16 个 attention head。序列 93k token，每卡持全序列 KV cache（MQA 共享，不切）。

```
每卡（16 heads）独立跑一遍 indexer + sparse attn:
┌──────────────────────────────────────────────────────────┐
│ Step 0  indexer 投影 (ReplicatedLinear,每卡完整权重)     │ ← 8 卡各算一遍,无切分无通信
│           wq_b / wk / weights_proj                       │
│ Step 1  scoring   fp8_mqa_logits [93k,93k]               │ ← 8 卡各读全 KV + 各算一遍
│ Step 2  topk      选 512                                │ ← 8 卡各算一遍
│ Step 3  KV gather  读 93k×512×576                        │ ← 8 卡各读全 KV
│ Step 4  attention  16 heads × 512 KV 算加权和            │ ← ★ TP 真切:每卡只 16/128 头
│ Step 5  o_proj + AllReduce                               │ ← ★ 跨 8 卡 AllReduce(唯一通信)
└──────────────────────────────────────────────────────────┘
       通信:仅 1 次 AllReduce(末尾)
       浪费:Step 0-3 八卡重复算/读 8 次
```

逐条确认（代码已查）：

| 步 | 并行策略 | 切没切 |
|---|---|---|
| 0 indexer proj | `ReplicatedLinear` → 每卡完整权重 | ❌ 每卡算同样 |
| 1 scoring | `fp8_mqa_logits` MQA 单头 | ❌ 每卡读全 KV、算同样 `[93k,93k]` |
| 2 topk | 算同样的 512 | ❌ 每卡同样 |
| 3 KV gather | 共享 576 latent，每卡全 KV cache 副本 | ❌ 每卡读全 |
| 4 attention 算术 | `q_b_proj`/`kv_b_proj` ColumnParallel → 16 heads | ✅ 唯一真切的 |
| 5 o_proj | RowParallel + AllReduce | ✅ 切 + 通信 |

**对 DSA attention 而言，TP 基本等于没切** —— 八卡把 indexer scoring 和 KV gather 重复跑 8 遍，只切了非瓶颈的算术部分（Step 4）。TP 还在用是因为 MoE 和 attention 的 compute 部分（Step 4）需要它，**不是 attention 的 memory 瓶颈需要它**。

---

## 5. CP=8 每卡执行流程

CP 开启时 `attn_tp_size=1`（[`parallel_state.py:1931`](../python/sglang/srt/distributed/parallel_state.py#L1931)，`attn_cp_size=8` → `attn_tp = 8//8 = 1`），attention head **不切**，每卡持完整 128 head。CP 切的是 `q_len`（序列维）。

```
每卡（128 heads,不切）执行:
┌──────────────────────────────────────────────────────────┐
│ [前] KV AllGather  (rebuild_cp_kv_cache)                 │ ← CP 通信: gather 全 KV
│ Step 0  indexer 投影 (每卡完整)                          │ ← q_len 11.6k(切了)
│ Step 1  scoring   [11.6k, 93k]                           │ ← q_len/8,KV 全(AllGather 来)
│ Step 2  topk      选 512                                 │ ← q_len/8
│ Step 3  KV gather 读 11.6k×512×576                     │ ← q_len/8
│ Step 4  attention 128 heads × 512 KV                    │ ← q_len/8(头不切但 seq 切)
│ [后] 输出 AllGather + rerange (cp_all_gather_rerange)    │ ← CP 通信
└──────────────────────────────────────────────────────────┘
       通信:KV AllGather + 输出 gather(2 次)
       无冗余:Step 0-3 每卡只处理自己的 11.6k
```

CP 每卡只处理 11.6k 个 Q token（全 93k 切 8 份），①②③ 全部按 `q_len` 砍 8×。KV 全长不变（AllGather 补齐），所以是 8× 不是 64×。

---

## 6. 7.6× 加速的来源

| 步 | 成本 ∝ | no-CP TP=8（每卡） | CP=8（每卡） |
|---|---|---|---|
| ① scoring + topk | `q_len × seq_len × 64` | `93k × 93k`（没切） | `11.6k × 93k` → /8 ✅ |
| ② KV gather | `q_len × 512 × 576` | `93k × 512 × 576`（没切） | /8 ✅ |
| ③ attention 算术 | `q_len × 128 × 512 × v` | TP 切 head 8× | q_len/8 |
| 通信 | — | AllReduce ×1（末尾） | KV AllGather + 输出 gather（~8.9%） |

`31.3s → 4.1s ≈ 7.6×` 的来源：**①② 从没切变成切 8×**。③ 两边都能切（TP 切 head、CP 切 q_len），不是差距来源。CP 代价是 KV AllGather（~8.9%），故实际 7.6× 而非 8×。

---

## 7. 对比表：TP vs CP 切到了什么

```
                    q_len 维        head 维        KV cache
─────────────────────────────────────────────────────────────
no-CP TP=8:   93k(全)        128/8=16       全 576 每卡复制  ← ①② 没切,瓶颈
CP=8:         93k/8=11.6k    128(不切)      全 576 AllGather  ← ①②③ 全切
```

| | no-CP TP=8 | CP=8 |
|---|---|---|
| 每卡 q_len | ~93k | ~11.6k |
| sparse_attn 每卡 | 31.3s | 4.1s |
| ① scoring | `93k × 93k` | `11.6k × 93k` → /8 |
| ② gather 读量 | `93k × 512 × 576` | /8 |
| ③ 算术 | TP 已切 head | q_len/8 |
| 通信 | AllReduce ×1 | KV AllGather + 输出 gather |

---

## 8. 常见误解纠正

### 8.1 "理论 64×" 是错的

有人算 `O(seq²)` → `(93000)² / (11600)² = 64×`，实测只有 7.6×，差距归因于 sparse attention 复杂度不到 `O(n²)`。**这个解释方向对，但 64× 的设定是错的。**

代码事实：DSA 是**稀疏** attention，KV 在 CP 下 **AllGather 到全长**（[`cp_utils.py:310`](../python/sglang/srt/layers/utils/cp_utils.py#L310) `rebuild_cp_kv_cache`），`kv_len` 不下降，只有 `q_len` 下降。scoring 成本 = `(seq/8) × seq` = `seq²/8` → **8×，不是 64×**。实测 7.6× 与 8× 吻合。

### 8.2 命名：是 DSA 不是 NSA

GLM 5.2 用的是 **DSA**（DeepSeek Sparse Attention，[`dsa/utils.py`](../python/sglang/srt/layers/attention/dsa/utils.py)），不是 NSA。`--enable-nsa-prefill-context-parallel` / `--nsa-prefill-cp-mode` 是已废弃别名（[`server_args.py:6820`](../python/sglang/srt/server_args.py#L6820) `DeprecatedStoreTrueAction`）。

### 8.3 "TP MoE 本地算完就结束" 是错的

（此为 RR vs DeepEP MoE 对比中的误解，非本文主题，但相关。）

RR MoE = TP（`moe_tp_size=8`），被 AllGather（前）+ ReduceScatter（后）包围，每层 3 次 NCCL 同步（KV AG + MLP AG + MLP RS）。**并非"本地算完就结束"**。详见 [`dsa_prefill_cp_glm_flow.md`](dsa_prefill_cp_glm_flow.md) 附录 A。

---

## 9. 一句话总结

**CP 比 TP 在 attention 上快 7.6×，因为 DSA attention 的瓶颈是 memory-bound 的 indexer scoring（MQA 单头）+ KV gather（MLA 共享 latent），二者都 ∝ `q_len`、与 128 个 attention head 无关。** TP 切 128 head 只动到非瓶颈的算术部分（Step 4），八卡把 ①② 重复跑 8 遍；CP 切 `q_len` 直接砍瓶颈 8×，代价是一次 KV AllGather（~8.9%）。MLA 的共享 latent + MQA indexer 本是为省 KV 带宽，副作用是让 TP 在 attention 上失效——正好被 CP 的 seq 切分治掉。

---

## 附录 A：关键代码索引

| 函数 / 符号 | 位置 | 作用 |
|---|---|---|
| `fp8_mqa_logits` | [`dsa_indexer.py:837`](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L837) | Step 1 scoring（MQA 单头，`[q_len, seq_len]`） |
| `index_score.topk` | [`dsa_indexer.py:1119`](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1119) | Step 2 选 topk=512 |
| `flash_mla_sparse_fwd` | [`dsa_backend.py:1779`](../python/sglang/srt/layers/attention/dsa/dsa_backend.py#L1779) | Step 3+4 sparse gather + attention |
| `DSAIndexer.__init__` | [`dsa_indexer.py:346`](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L346) | indexer 投影层（`ReplicatedLinear`，不切） |
| `MLATokenToKVPool.kv_buffer` | [`memory_pool.py:1805`](../python/sglang/srt/mem_cache/memory_pool.py#L1805) | MLA KV cache 存储（`[size,1,576]`，71× 压缩） |
| `DeepseekV2AttentionMLA.__init__` | [`deepseek_v2.py:1467`](../python/sglang/srt/models/deepseek_v2.py#L1467) | MLA attention 层（`num_local_heads = num_heads // attn_tp_size`） |
| `q_b_proj` / `kv_b_proj` / `o_proj` | [`deepseek_v2.py:1500`](../python/sglang/srt/models/deepseek_v2.py#L1500) / [`:1570`](../python/sglang/srt/models/deepseek_v2.py#L1570) / [`:1580`](../python/sglang/srt/models/deepseek_v2.py#L1580) | TP 切 head 的唯一三处（ColumnParallel / RowParallel） |
| `rebuild_cp_kv_cache` | [`deepseek_v2.py:1874`](../python/sglang/srt/models/deepseek_v2.py#L1874) | CP 下 KV AllGather（两模式共用） |
| `cp_all_gather_rerange_output` | [`cp_utils.py:310`](../python/sglang/srt/layers/utils/cp_utils.py#L310) | AllGather + 恢复顺序 |
| `attn_tp_size` 推导 | [`parallel_state.py:1931`](../python/sglang/srt/distributed/parallel_state.py#L1931) | CP 开启 → `attn_tp_size=1`（head 不切） |
| `is_deepseek_dsa` | [`model_config.py:102`](../python/sglang/srt/configs/model_config.py#L102) | 判 DSA 架构（需 `index_topk`） |
| `index_topk` / `index_n_heads` | [`deepseek_v4.py:60`](../python/sglang/srt/configs/deepseek_v4.py#L60) / [`:59`](../python/sglang/srt/configs/deepseek_v4.py#L59) | DSA indexer 参数（512 / 64） |

## 附录 B：维度速查（GLM 5.2 DSA）

| 维度 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 7168 | 隐藏维度 |
| `num_attention_heads` | 128 | attention head 数（TP 切这个） |
| `num_local_heads`（TP=8） | 16 | 每卡 attention head 数 |
| `kv_lora_rank` | 512 | MLA 压缩 K 维 |
| `qk_rope_head_dim` | 64 | rope 维 |
| `qk_nope_head_dim` | 128 | nope 维 |
| `kv_cache_dim` | 576 | `kv_lora + qk_rope` = 512+64，全 head 共享 |
| 展开等价（不存） | 40960 | `128 × (128+128+64)` |
| 压缩比 | 71× | `40960 / 576` |
| `index_topk` | 512 | 每 Q token 挑中的 KV 页数 |
| `index_n_heads` | 64 | indexer 小头数（非 128 attention head） |
| `index_head_dim` | 128 | indexer head 维 |
| `page_size` | 64 | KV cache 分页大小 |

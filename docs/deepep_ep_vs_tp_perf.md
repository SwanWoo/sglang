# round-robin-split（TP）vs in-seq-split（DeepEP）MoE 性能对比

> 本文档记录在 DeepSeek 系 MoE 模型上，DSA Prefill Context Parallel 两种 split 模式的性能对比：
> - `round-robin-split` → 自动走 **TP-only MoE**（`moe_a2a_backend=none`，`ep_size=1`）
> - `in-seq-split` → 自动强制 **DeepEP（EP + All-to-All）**（`moe_a2a_backend=deepep`，`ep_size=tp_size`）
>
> 两种 split 模式与 MoE 后端在启动参数解析阶段绑定（[server_args.py:1861-1875](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1861-L1875)），无法解耦，故本测既是 split 模式对比，也是 TP-vs-DeepEP 后端对比。
> 并从代码层面解释 MoE / EP / DeepEP 的原理，以及**为何稳态大 batch prefill 下 in-seq-split（DeepEP）反而慢于 round-robin-split（TP）**（[第 2 节](#2-性能测试结果)）。
>
> 相关文档：
> - DSA Prefill CP 完整流程：[`dsa_prefill_cp_round_robin_split_flow.md`](dsa_prefill_cp_round_robin_split_flow.md)
> - DSA Prefill CP 数学原理：[`dsa_prefill_cp_round_robin_math.md`](dsa_prefill_cp_round_robin_math.md)
>
> 本文原理部分基于以下代码：
> - DeepEP dispatcher：[`python/sglang/srt/layers/moe/token_dispatcher/deepep.py`](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py)
> - DeepEP 模式枚举与解析：[`python/sglang/srt/layers/moe/utils.py`](../python/sglang/srt/layers/moe/utils.py)
> - EP MoE 层：[`python/sglang/srt/layers/moe/ep_moe/layer.py`](../python/sglang/srt/layers/moe/ep_moe/layer.py)
> - MoE TP/EP 切分计算：[`python/sglang/srt/distributed/parallel_state.py`](../python/sglang/srt/distributed/parallel_state.py)
> - 启动参数与自动配置：[`python/sglang/srt/server_args.py`](../python/sglang/srt/server_args.py)

---

## 1. 测试环境与启动参数

测试在单机 8 GPU 上运行 DeepSeek 系 MoE 模型，启用 DSA Prefill Context Parallel，对比两种 CP split 模式：`round-robin-split`（TP-only MoE）vs `in-seq-split`（自动开启 EP + DeepEP）。

### 1.1 Baseline：无 DeepEP（TP-only MoE）

```bash
python -m sglang.launch_server \
  --model-path /models \
  --trust-remote-code \
  --port 30000 \
  --mem-fraction-static 0.8 \
  --tp-size 8 \
  --enable-nsa-prefill-context-parallel \
  --attn-cp-size 8 \
  --nsa-prefill-cp-mode round-robin-split \
  --disable-radix-cache
```

> 注：`--enable-nsa-prefill-context-parallel` / `--nsa-prefill-cp-mode` 为已废弃别名
> （`DeprecatedStoreTrueAction`，[server_args.py:6820](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L6820) / `DeprecatedAliasStoreAction`，[server_args.py:6835](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L6835)），
> 当前推荐使用 `--enable-dsa-prefill-context-parallel` / `--dsa-prefill-cp-mode round-robin-split`。

关键点：未传 `--moe-a2a-backend` 与 `--ep-size`。`moe_a2a_backend` 默认 `"none"`（[server_args.py:636](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L636)），`ep_size` 默认 `1`（[server_args.py:635](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L635)）。
round-robin-split 的初始化路径（[server_args.py:1870-1875](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1870-L1875)）只设置 `enable_dp_attention = True` 和 `moe_dense_tp_size = 1`，**不会自动开启 DeepEP / EP**，
因此 baseline 走的是 TP 权重切分的 non-a2a 路径（详见 [第 5 节](#5-tp-模式-moebaseline-路径)）。

### 1.2 对照组：开启 DeepEP（EP + All-to-All）

```bash
python -m sglang.launch_server \
  --model-path /models \
  --trust-remote-code \
  --port 30000 \
  --mem-fraction-static 0.8 \
  --tp-size 8 \
  --enable-nsa-prefill-context-parallel \
  --attn-cp-size 8 \
  --nsa-prefill-cp-mode in-seq-split \
  --disable-radix-cache
```

与 baseline 唯一差异：`--nsa-prefill-cp-mode in-seq-split`（替代 `round-robin-split`）。

`in-seq-split` 模式在启动参数解析阶段自动强制开启 DeepEP（[server_args.py:1861-1868](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1861-L1868)）：

```python
if self.dsa_prefill_cp_mode == "in-seq-split":
    self.enable_dp_attention = True
    self.moe_dense_tp_size = 1
    self.moe_a2a_backend = "deepep"   # 自动强制 DeepEP
    self.ep_size = self.tp_size        # 自动强制 ep_size = tp_size = 8
```

即无需手动传 `--moe-a2a-backend deepep --ep-size 8 --deepep-mode auto`，`in-seq-split` 已隐含等价配置（`--deepep-mode` 默认即 `auto`，[server_args.py](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py)）。因此对照组相比 baseline 的净变量 = **CP split 模式（round-robin → in-seq）+ 隐式开启 EP/DeepEP**，两者绑定为同一变量。

> ⚠️ **in-seq-split 限制**：启动日志会 warning「`moe_dense_tp_size == 1, moe_a2a_backend == deepep, ep_size == tp_size, batch_size == 1`」（[server_args.py:1868](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1868)）。`batch_size == 1` 指 extend batch 单序列约束，benchmark 侧需保证每次 extend 单序列（多并发由 scheduler 拆分到逐序列 extend）。round-robin-split 无此约束。
>
> ⚠️ **纯净变量控制说明**：因 `in-seq-split` 自动绑定 EP/DeepEP，无法在 in-seq-split 下关闭 DeepEP 做「同 split 模式 ±EP」对比。故本测为 **round-robin-split(TP) vs in-seq-split(EP)** 的对比，CP split 模式本身差异（token 分布不同，见 [`dsa_prefill_cp_round_robin_math.md`](dsa_prefill_cp_round_robin_math.md) vs [`dsa_prefill_cp_in_seq_split_flow.md`](dsa_prefill_cp_in_seq_split_flow.md)）与 EP-vs-TP 差异耦合。但 attention 侧两者 `attn_tp_size=1` 相同（[第 7 节](#7-cp8-下-tp-与-ep-的并行度与通信对比)），MoE MLP 部分的 TP-vs-EP 是主导差异项。

---

## 2. 性能测试结果

### 2.1 吞吐与 TTFT 对比

> 两种 CP split 模式的对比：round-robin-split（TP-MoE，无 EP）vs in-seq-split（自动 EP + DeepEP）。
>
> **条件**：`--disable-radix-cache`（关闭 radix cache），Sonnet ~22,934 tokens in / 1 token out，64 并发，单机 8 GPU。

| CP Mode | EP | DeepEP | Run 1 | Run 2 | Run 3 | 最佳 tok/s | TTFT (Run3) |
|---------|:--:|:------:|------:|------:|------:|-----------:|------------:|
| round-robin-split | ❌ | ❌ | 83,310 | 89,505 | 89,402 | **89,505** | 8,447 ms |
| in-seq-split | ✅(auto) | ✅(auto) | 66,715 | 72,133 | 71,194 | **72,133** | 10,604 ms |

**结论（稳态）**：prefill-CP 大 batch 场景下，**EP-MoE MLP（in-seq-split + DeepEP）慢于 TP-MoE MLP（round-robin-split）约 19%**（72k vs 89k tok/s），TTFT 高 25%（10.6s vs 8.4s）。

### 2.2 结果解读

稳态下两种模式三次运行都稳定（round-robin 83k/89k/89k，in-seq 67k/72k/71k），说明都已 warm，排除了冷启动方差。**EP 反而慢**，与「EP 通信量少、可 overlap 故更快」的直觉相反。根因见 [第 8 节](#8-为什么稳态大-batch-prefill-下-in-seq-split-反而慢于-round-robin-split) 的分析：

1. **DeepEP normal 模式单层内 dispatch→GEMM→combine 是纯串行 critical path**，`async_finish=True` 只让通信落 comm stream，但 [`dispatch_b` 后立即 `event.current_stream_wait()`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L488)，compute stream 必须等 dispatch 完才能跑 expert GEMM。`dispatch_a/b` 拆分是为 two-batch overlap（跨层），单层内无 overlap。本次未开 TBO → 通信全程在 critical path。
2. **A2A 通信量被 top_k 放大**。DeepEP 模式 [`_is_deepep_fusion`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L543-L551) 把 shared expert 融进 MoE，top_k 8→9、expert 数 256→264，每 token 按 9 份发往 9 个 expert，两次 A2A（dispatch+combine）共 ~9× 放大。大 batch prefill（64 并发 × 22934 ≈ 1.47M token）下 A2A 总量压过 TP 的单次 AllReduce。
3. **EP 路径失去 shared expert dual-stream overlap**（⚠️ 默认 fusion-on 下不成立，见 [第 8.1 节 ③](#81-针对-dsa-prefill-cp-场景稳态下-ep-反而慢的两个原因) 修正）。TP 路径 [`forward_normal_dual_stream`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L863) 把 shared expert 放 `alt_stream` 与 routed expert 并行；但默认 `disable_shared_experts_fusion=False` 时 TP 走的是 `forward_normal`（fusion-on），两条路径 shared 都已融进 MoE kernel，无独立 shared expert 可 overlap（除非开 SBO 或 fusion-off）。

### 2.3 何时 EP 会赢

EP 的优势区不在大 batch prefill 稳态，而在：
- **小 batch / decode**：A2A 量小，pairwise 延迟低于 AllReduce 全员同步；low-latency 模式 + `return_recv_hook` 真 overlap。
- **长 context 但 token 总量小**（如单序列 128k）：TP AllReduce 量 = `hidden × tokens` 大，A2A 放大后仍可能更小。
- **开启 two-batch overlap**：跨层 overlap 抵消单层串行。

本次场景（64 并发 × 22934 = 1.47M token 大 batch prefill，TBO 未开）正好踩 EP 劣势区。

---

## 3. MoE 原理

MoE（Mixture of Experts）= 稀疏激活。每个 token 不走全部 expert，只走 top-k 个（DeepSeek 系 top-k 通常为 8，总 expert 数 256）。

### 3.1 单层 MoE 前向流程

```
hidden_states [N, D]
        │
        ▼
   ┌─────────┐
   │ Router  │  gate logits = W_gate @ x
   └────┬────┘
        │  fused_topk → 选 top-k expert + 权重
        ▼
   ┌─────────────────┐
   │ Token Dispatch  │  按 topk_ids 把 token 分发到对应 expert
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────────┐
   │ Expert 计算（per expert）   │  每个 expert = 两个 GEMM
   │  gate_up_proj → act → down  │  W1, W3 → SiLU → W2
   └────────────┬────────────────┘
                │
                ▼
   ┌─────────────────────┐
   │ Combine（加权求和） │  按 topk_weights 聚合各 expert 输出
   └──────────┬──────────┘
              ▼
      hidden_states [N, D]
```

代码层面（[`router.py`](../python/sglang/srt/layers/moe/router.py)）：
- `FusedMoeRouter.forward_cuda()` 调 `fused_moe_router_shim`（[router.py:340](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/router.py#L340)）
- 大 batch 或 expert 多时走 tensorcore 路径 `fused_moe_router_tensorcore`（[router.py:286](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/router.py#L286)），否则走 cudacore 路径
- `fused_topk` 选 top-k expert + renormalize 权重

### 3.2 MoE 的核心问题

256 个 expert 但每 token 只激活 8 个 → 计算量省，但**权重占显存大**。问题变成：
- 256 个 expert 的权重怎么放到 GPU 上？
- token 怎么送到它该去的 expert？

这是 TP vs EP 的分歧点（见 [第 5 节](#5-tp-模式-moebaseline-路径) 与 [第 6 节](#6-ep-原理与-deepep-实现)）。

---

## 4. 分布式集合通信原语

TP 和 EP 的核心区别在于使用了不同的集合通信原语。本节解释四种常见原语的语义与区别。

### 4.1 AllReduce

**语义**：每张卡对本地数据做一次 reduce（如 sum），再把结果 **广播给所有卡**，最终所有卡拿到**相同完整结果**。

```
输入（每卡各一份）：
  Rank 0: [1, 2]      Rank 1: [3, 4]
  Rank 2: [5, 6]      Rank 3: [7, 8]

AllReduce(sum) 后，所有卡得到：
  Rank 0: [16, 20]
  Rank 1: [16, 20]
  Rank 2: [16, 20]
  Rank 3: [16, 20]
```

- **数据量**：每卡发送/接收 `O(N)`（N = 张量大小），内部实现为 ReduceScatter + AllGather 组合
- **典型用途**：TP 模式下每层 MoE 后合并 partial result——每卡算 1/8 权重的部分输出，AllReduce 求和还原完整 hidden_states
- **瓶颈**：所有卡必须同步等待，延迟 = 最慢卡的完成时间 + 通信开销

### 4.2 AllGather

**语义**：每卡持有数据的一个**分片**，收集所有卡的分片**拼接**成完整张量，所有卡拿到**相同完整结果**。

```
输入（每卡持有一个分片）：
  Rank 0: [A]      Rank 1: [B]
  Rank 2: [C]      Rank 3: [D]

AllGather 后，所有卡得到：
  Rank 0: [A, B, C, D]
  Rank 1: [A, B, C, D]
  Rank 2: [A, B, C, D]
  Rank 3: [A, B, C, D]
```

- **数据量**：每卡发送本地分片 `O(N/P)`，接收所有分片 `O(N)`（P = 卡数，N = 完整张量大小）
- **典型用途**：CP 模式下 AllGather 收集各 CP rank 的 KV 分片，用于 attention 计算
- **与 AllReduce 的区别**：AllGather 是**拼接**（concatenate），AllReduce 是**归约**（sum/max/min 等）

### 4.3 ReduceScatter

**语义**：对所有卡的数据做 reduce（如 sum），但结果**不广播**，而是按分片分散到对应卡——每卡只拿到**属于自己的那一份 reduce 结果**。

```
输入（每卡各一份完整数据）：
  Rank 0: [1, 2]      Rank 1: [3, 4]
  Rank 2: [5, 6]      Rank 3: [7, 8]

ReduceScatter(sum) 后，每卡只拿到一个分片：
  Rank 0: [16]   ← 所有卡第 0 分片的 sum
  Rank 1: [20]   ← 所有卡第 1 分片的 sum
  Rank 2: [24]   ← 所有卡第 2 分片的 sum
  Rank 3: [28]   ← 所有卡第 3 分片的 sum
```

- **数据量**：每卡发送 `O(N)`，只接收 `O(N/P)` —— 比 AllReduce 节省一半通信（AllReduce = ReduceScatter + AllGather）
- **典型用途**：CP 模式下 ReduceScatter 把 MoE 层输出按 CP rank 切分，避免后续 AllGather 前再做一次完整广播
- **与 AllGather 的关系**：ReduceScatter 是 AllGather 的逆操作。`AllReduce = ReduceScatter → AllGather`

### 4.4 All-to-All（A2A）

**语义**：每卡把本地数据按**目标 rank 分组**，发给对应卡；同时接收其他卡发来的数据。每卡发送和接收的数据量由实际路由决定，**不需要所有卡数据量相同**。

```
输入（每卡持有发给各 rank 的分组数据）：
  Rank 0: [→R0: a,  →R1: b,  →R2: c,  →R3: d ]
  Rank 1: [→R0: e,  →R1: f,  →R2: g,  →R3: h ]
  Rank 2: [→R0: i,  →R1: j,  →R2: k,  →R3: l ]
  Rank 3: [→R0: m,  →R1: n,  →R2: o,  →R3: p ]

All-to-All 后，每卡收到发给自己的分组：
  Rank 0: [a, e, i, m]   ← 所有卡发给 R0 的数据
  Rank 1: [b, f, j, n]   ← 所有卡发给 R1 的数据
  Rank 2: [c, g, k, o]   ← 所有卡发给 R2 的数据
  Rank 3: [d, h, l, p]   ← 所有卡发给 R3 的数据
```

- **数据量**：每卡发送 `O(N)`、接收 `O(N)`，但通信模式是 **pairwise 定向传输**，不是全员同步
- **典型用途**：EP 模式下 token 按路由结果发到对应 expert 所在的卡（dispatch），计算完再发回来（combine）
- **与 AllReduce/AllGather/ReduceScatter 的核心区别**：
  - 前三者是**结构化通信**——每卡发送/接收的数据位置和大小由算法固定（ring、tree 等）
  - A2A 是**数据驱动通信**——每条消息的目标由数据内容决定（哪个 token 去哪个 expert），大小不均匀
  - A2A 天然适合稀疏激活场景（MoE：每 token 只去少数 expert），不需要全员同步

### 4.5 四种原语对比

| 原语 | 语义 | 输入 | 输出 | 通信模式 | 典型场景 |
|------|------|------|------|----------|----------|
| **AllReduce** | 归约 + 广播 | 每卡完整张量 | 每卡相同完整结果 | 全员同步 | TP partial result 合并 |
| **AllGather** | 收集 + 拼接 | 每卡一个分片 | 每卡相同完整结果 | 全员同步 | CP KV 分片收集 |
| **ReduceScatter** | 归约 + 分散 | 每卡完整张量 | 每卡一个分片 | 全员同步 | CP 输出按 rank 切分 |
| **All-to-All** | 定向传输 | 每卡按目标分组 | 每卡收到发给自己 | Pairwise 定向 | EP token dispatch/combine |

关键维度：

- **同步性**：AllReduce / AllGather / ReduceScatter 都是**全员同步**——所有卡必须到齐才能完成；All-to-All 是 **pairwise**——只需发送方和接收方就绪
- **数据感知**：前三种通信模式与数据内容无关（固定 pattern）；All-to-All 通信内容由数据决定（哪个 token 去哪张卡）
- **通信量**：AllReduce ≈ ReduceScatter + AllGather；All-to-All 通信量取决于实际路由，通常更少（稀疏激活）
- **可 overlap 性**：All-to-All 天然可拆分（dispatch/combine 理论上可与计算 overlap）；AllReduce 必须等所有 partial 完成才能 reduce，纯串行。**但 DeepEP normal 模式实现里 `dispatch_b` 后立即 `current_stream_wait`，单层内实际无 overlap**（[第 8.1 节](#81-针对-dsa-prefill-cp-场景稳态下-ep-反而慢的两个原因)），仅跨层 TBO 可 overlap。

> 在本文的 split 模式对比中：round-robin-split（TP）用 AllReduce 合并 partial result（全员同步、纯串行），in-seq-split（EP）用两次 All-to-All 做 dispatch + combine（定向传输、理论上可 overlap）。**但稳态大 batch prefill 下 EP 反而慢**——normal 模式单层内 A2A 串行无 overlap + top_k 放大通信量，通信劣势压过定向传输优势（详见 [第 8 节](#8-为什么稳态大-batch-prefill-下-in-seq-split-反而慢于-round-robin-split)）。

### 4.6 集合通信的底层算法：Ring / Tree / NVLS / CollNet

上层的 AllReduce / AllGather / ReduceScatter 是「语义原语」，但同一次 AllReduce 在 NCCL 底层可用不同**算法**实现。本节解释四种常见底层算法，并说明本代码库实际用的是哪种。

#### 4.6.1 Ring（环）

**结构**：把 P 张卡排成一个环，每张卡只与左右邻居通信。数据切成 P 份，分 P-1 步流转，每步每卡发给右邻一份、收左邻一份并在本地 reduce。

```
AllReduce(Ring) 示意（P=4，切 4 份 a/b/c/d）：

Step 0   R0→R1:a   R1→R2:b   R2→R3:c   R3→R0:d
Step 1   R0→R1:d'  R1→R2:a'  R2→R3:b'  R3→R0:c'   (累计 reduce)
  ...    共 P-1 = 3 步完成 reduce，再 P-1 步 AllGather 分发结果
```

- **带宽利用率**：理论近 100%——每步每卡都在发/收，链路充分利用
- **延迟**：`O(P)` 步，P 大时步数线性增长
- **适用**：中等规模、带宽敏感场景。**NCCL 默认 AllGather 算法就是 Ring**，单机 NVLink 上表现稳定
- **缺点**：P 很大时延迟随 P 增长；依赖环不断——任一卡慢，全环卡住（木桶效应）

#### 4.6.2 Tree（树）

**结构**：把卡组织成树（常见二叉树或 k 叉树）。Reduce 阶段从叶子向根聚合，Broadcast 阶段从根向叶子分发。

```
        Root
       / | \
     R0  R1  R2      ← Reduce: 叶→根聚合
      \  |  /
        Root         ← Broadcast: 根→叶分发
```

- **延迟**：`O(log P)` 步——比 Ring 的 `O(P)` 对数级更优，P 大时延迟低
- **带宽利用率**：低——每步只有树的某一层在通信，大部分链路空闲
- **适用**：大规模、延迟敏感（跨机多机）、小消息。NCCL 对跨机节点间常选 Tree
- **缺点**：根节点成为热点，带宽利用率远低于 Ring

#### 4.6.3 Ring vs Tree 权衡

| 维度 | Ring | Tree |
|------|------|------|
| 步数（延迟） | `O(P)` | `O(log P)` |
| 带宽利用率 | 高（~100%） | 低（每步一层） |
| 大消息 | ✅ 优（带宽为主） | ❌ 差 |
| 小消息 / 大 P | ❌ 延迟随 P 涨 | ✅ 对数延迟 |
| 热点 | 无 | 根节点 |

NCCL 运行时按**消息大小 + 拓扑 + P** 自动选 Ring 或 Tree（也可用 `NCCL_ALGO`/`NCCL_PROTOCOL` 强制）。大消息（本文 1.47M token prefill）默认 Ring。

#### 4.6.4 NVLS / CollNet（NVLink SHARP）

**NVLS（NVLink SHARP）**：Hopper/Blackwell + NVSwitch 拓扑下，利用 NVSwitch 硬件 in-network reduce（交换机内做加法），AllReduce 可一步完成，跳过 Ring 的 P-1 步。

- **前提**：NVSwitch（NVL8/NVL32 等）+ NCCL ≥ 2.18 + 拓扑支持
- **带宽**：可超 Ring（硬件 reduce，无多次跳转）
- **限制**：仅同机 NVSwitch 域内；消息需对齐；部分大小自动 fallback 到 Ring
- **CollNet**：SHARP 的多机扩展（InfiniBand 交换机 in-network reduce）

```
普通 Ring：R0→R1→R2→R3→...→R0   P-1 步 reduce
NVLS：    各卡 → NVSwitch（硬件 reduce）→ 各卡   1 步
```

#### 4.6.5 本代码库实际用哪种 —— AllGather 是否用 Ring？

本代码库集合通信最终落到 [pynccl.py](../python/sglang/srt/distributed/device_communicators/pynccl.py) 的裸 NCCL 调用：

```python
# AllGather（pynccl.py:216）
self.nccl.ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream)
# AllReduce（pynccl.py:143）
self.nccl.ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
# ReduceScatter（pynccl.py:289）
self.nccl.ncclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm, stream)
```

**关键事实：代码没有传 `algo` 参数，全部交给 NCCL 运行时自动选算法。** 因此：

1. **AllGather 默认走 Ring**（NCCL AllGather 的默认算法）。TP/CP 路径的 `attn_cp_all_gather_into_tensor`（[dp_attention.py:599](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L599)）→ `GroupCoordinator._all_gather_into_tensor`（[parallel_state.py:821](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L821)）→ `pynccl.all_gather` → `ncclAllGather`，无 algo pin，NCCL 在单机 NVLink + 大消息下默认选 **Ring**（若 NVSwitch 拓扑满足且 `NCCL_ALGO` 未禁，可能自动升 NVLS）。
2. **AllReduce 同理**：`tensor_model_parallel_all_reduce`（[deepseek_v2.py:1034](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1034)）→ in-place `pynccl.all_reduce` → `ncclAllReduce`，NCCL 自动选 Ring（大消息）/ Tree（小消息或跨机）。
3. **另有自定义 in-network 旁路**（非 NCCL）：
   - `ca_comm.custom_all_reduce`（custom all-reduce，[parallel_state.py:727](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L727)）：SGLang 自研低延迟 AllReduce，小 batch decode 走此旁路绕过 NCCL
   - `torch_symm_mem_comm.all_reduce`（symmetric memory，[parallel_state.py:733](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L733)）：基于 NVLink symmetric memory 的 AllReduce
   - `qr_comm.quick_all_reduce` / `pymscclpp_comm`：其他旁路
   - 但这些主要用于 out-place AllReduce（decode 小 batch 路径）。**本文大 batch prefill 的 MoE AllReduce 走的是 in-place `pynccl.all_reduce` = NCCL Ring/Tree**，非自定义旁路。

4. **DeepEP A2A 不是 NCCL 集合通信**：DeepEP 用自研 RDMA+NVLink 拓扑感知 kernel（[`DeepEPBuffer`](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py)），不走 `ncclAllToAll`，因此 Ring/Tree/NVLS 分类不适用——它是 pairwise 定向传输，算法由 DeepEP buffer 内部按 `num_tokens_per_rdma_rank`（跨机走 RDMA）与 `num_tokens_per_rank`（同机走 NVLink）分流决定。

> **结论**：本文 TP 路径的 AllGather / AllReduce / ReduceScatter 在代码层面不显式选 Ring，但 NCCL 默认对大消息选 **Ring**（NVSwitch 拓扑满足时可能 NVLS）。EP 路径的 A2A 不属于 Ring/Tree 范畴，是 DeepEP 自研 pairwise kernel。可用 nsys profile 中 `ncclAllGather`/`ncclAllReduce` 的 algo 字段（或设 `NCCL_DEBUG=INFO`）确认运行时实际选的算法。

### 4.7 各原语在 sglang 代码中的实际使用

本节给出四种原语在代码层面的调用链与关键 call site。所有 NCCL 原语最终落到 [`pynccl.py`](../python/sglang/srt/distributed/device_communicators/pynccl.py) 的裸 NCCL 调用（无 algo pin，算法由 NCCL 运行时选，见 [第 4.6.5 节](#465-本代码库实际用哪种--allgather-是否用-ring)）；All-to-All 例外，走 DeepEP 自研 kernel。

#### 4.7.1 AllReduce

**调用链**：

```
model 代码 (e.g. deepseek_v2.py:1034)
  └─ tensor_model_parallel_all_reduce(input_)          communication_op.py:18
       └─ get_tp_group().all_reduce(input_)            parallel_state.py:562
            ├─ _all_reduce_in_place(input_)            parallel_state.py:743   (in-place, 默认)
            │    └─ pynccl_comm.all_reduce(input_)     pynccl.py:132
            │         └─ ncclAllReduce(...)            pynccl.py:143  ← NCCL Ring/Tree
            └─ _all_reduce_out_place(input_, method)   parallel_state.py:716   (out-place, decode 小 batch)
                 ├─ ca_comm.custom_all_reduce          (SGLang 自研低延迟 AR)
                 ├─ qr_comm.quick_all_reduce
                 ├─ torch_symm_mem_comm.all_reduce     (NVLink symmetric memory)
                 ├─ pymscclpp_comm.all_reduce
                 └─ pynccl_comm.outplace_all_reduce    pynccl.py:153 → ncclAllReduce
```

**关键 call site**：

| 位置 | 用途 | 走哪条 |
|------|------|--------|
| [`deepseek_v2.py:1034`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1034) / [:920](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L920) / [:1094](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1094) | TP MoE post-experts 合并 partial result | in-place pynccl |
| [`fused_moe_triton/layer.py:1113-1114`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1113-L1114) | `FusedMoE.forward_impl` 末尾，`moe_tp_size>1 or moe_ep_size>1` 时 | in-place pynccl |
| [`dp_attention.py:591`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L591) `attn_tp_all_reduce` | attention TP partial o_proj 合并 | attn_tp_group |
| `o_proj` / `gate_up_proj` / `down_proj`（`RowParallelLinear`） | 线性层 TP 后合并 | in-place，decode 小 batch 可能走 custom AR |

> 本文 round-robin-split(TP) 路径每层 MoE 的 AllReduce 即 `deepseek_v2.py:1034`，走 in-place `ncclAllReduce`。EP 路径 `moe_tp_size=1` 且 `forward_deepep` 不调此 AllReduce（见 [§4.7.4](#474-all-to-alla2a)）。

#### 4.7.2 AllGather

**调用链**：

```
dsa_cp_gather_hidden_states(hidden_states)            communicator_dsa_cp.py:55
  └─ attn_cp_all_gather_into_tensor(out, in)          dp_attention.py:599
       └─ get_attention_cp_group().all_gather_into_tensor(out, in)
            └─ reg_all_gather_into_tensor(out, in, group_name)   parallel_state.py:178 (custom op)
                 └─ GroupCoordinator._all_gather_into_tensor     parallel_state.py:821
                      ├─ (ROCm) ca_comm.all_gather_reg/unreg     (aiter 自定义)
                      └─ (CUDA) pynccl_comm.all_gather(out, in)  pynccl.py:181
                           └─ ncclAllGather(...)                 pynccl.py:216  ← NCCL Ring/NVLS
```

**关键 call site**：

| 位置 | 用途 |
|------|------|
| [`communicator_dsa_cp.py:63`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L63) `dsa_cp_gather_hidden_states` | **CP MLP 前 AllGather**——跨 CP rank 收集 hidden_states，TP 路径每层 1 次 |
| [`dp_attention.py:595`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L595) `attn_tp_all_gather_into_tensor` | attention TP 侧 AllGather |
| KV AllGather（attention 内部） | 跨 CP rank 收集 KV 分片，两种路径每层都 1 次 |

> `dsa_cp_gather_hidden_states` 先把本地 `local_hidden_states` 写入 `get_local_dp_buffer`，再 `attn_cp_all_gather_into_tensor` 拼成完整 token（[communicator_dsa_cp.py:59-64](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L59-L64)）。EP 路径 `mlp_mode=SCATTERED`，此函数走 `_simple` 旁路不实际 gather（[communicator_dsa_cp.py:132-133](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L132-L133)）。

#### 4.7.3 ReduceScatter

**调用链**：

```
dsa_cp_reduce_scatter_hidden_states(hidden_states)    communicator_dsa_cp.py:67
  └─ hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]   (本地先切片)
  └─ attn_cp_reduce_scatter_tensor(out, in)           dp_attention.py:587
       └─ get_attention_cp_group().reduce_scatter_tensor(out, in)
            └─ reg_reduce_scatter_tensor(out, in, group_name)  parallel_state.py:189
                 └─ GroupCoordinator._reduce_scatter_tensor   parallel_state.py:753
                      └─ pynccl_comm.reduce_scatter(out, in)   pynccl.py:252
                           └─ ncclReduceScatter(...)           pynccl.py:289  ← NCCL Ring/NVLS
```

**关键 call site**：

| 位置 | 用途 |
|------|------|
| [`communicator_dsa_cp.py:75`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L75) `dsa_cp_reduce_scatter_hidden_states` | **CP MLP 后 ReduceScatter**——MoE 输出按 CP rank 切分 + reduce，TP 路径每层 1 次 |

> ReduceScatter = AllReduce 的「只 reduce 不广播」版。`dsa_cp_reduce_scatter_hidden_states` 先 `tensor_split` 取本 rank 分片，再 `ncclReduceScatter` 跨 rank 求和。EP 路径同样 `mlp_mode=SCATTERED` 旁路，不调用。

#### 4.7.4 All-to-All（A2A）

**调用链**（不走 NCCL，DeepEP 自研）：

```
FusedMoE.forward_impl(hidden_states, topk_output)     fused_moe_triton/layer.py:1091
  ├─ dispatcher.dispatch(hidden_states, topk_output)  layer.py:1095
  │    └─ DeepEPDispatcher.dispatch_b()               deepep.py:897
  │         └─ _DeepEPDispatcherImplNormal.dispatch_b  deepep.py:480
  │              └─ _dispatch_core(x, topk_ids, ...)   deepep.py:503
  │                   └─ buffer.dispatch(...)          deepep.py:487  ← DeepEP RDMA+NVLink kernel
  │                        (按 num_tokens_per_rdma_rank 走 RDMA 跨机,
  │                         num_tokens_per_rank 走 NVLink 同机)
  ├─ run_moe_core(dispatch_output)                    layer.py:1099  (expert GEMM)
  └─ dispatcher.combine(combine_input)                layer.py:1106
       └─ DeepEPDispatcher.combine_b()                deepep.py:908/580
            └─ _combine_core(x, previous_event)       deepep.py:587
                 └─ buffer.combine(x, self.handle)    ← 反向 A2A, 结果发回原卡
```

**关键事实**：

- DeepEP **不调** `torch.distributed.all_to_all` / `ncclAllToAll`，而是 [`DeepEPBuffer`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L150) 自研 RDMA + NVLink 拓扑感知 kernel（[`get_dispatch_layout`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L503) 算 `num_tokens_per_rdma_rank` / `num_tokens_per_rank` 分流）。
- 两次 A2A：`dispatch`（token → expert 所在卡）+ `combine`（结果 → 原卡），替代 TP 路径的 AllReduce + CP AllGather + ReduceScatter。
- `_dispatch_core` / `_combine_core` 的 `async_finish=True` 让通信落 comm stream，但 `dispatch_b`/`combine_b` 末尾 `event.current_stream_wait()`（[deepep.py:488](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L488) / [:582](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L582)）把 compute stream 拉回等待——单层内串行，无 overlap（见 [第 8.1 节 ①](#81-针对-dsa-prefill-cp-场景稳态下-ep-反而慢的两个原因)）。

#### 4.7.5 四原语代码层总览

| 原语 | 顶层 wrapper | GroupCoordinator 方法 | 底层 kernel | 本文 call site |
|------|-------------|----------------------|------------|---------------|
| **AllReduce** | `tensor_model_parallel_all_reduce` ([communication_op.py:18](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/communication_op.py#L18)) | `all_reduce` / `_all_reduce_in_place` ([ps:562](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L562)/[:743](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L743)) | `ncclAllReduce` ([pynccl.py:143](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/device_communicators/pynccl.py#L143)) / custom AR | `deepseek_v2.py:1034`（TP MoE） |
| **AllGather** | `attn_cp_all_gather_into_tensor` ([dp_attention.py:599](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L599)) | `_all_gather_into_tensor` ([ps:821](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L821)) | `ncclAllGather` ([pynccl.py:216](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/device_communicators/pynccl.py#L216)) / aiter | `communicator_dsa_cp.py:63`（CP MLP 前） |
| **ReduceScatter** | `attn_cp_reduce_scatter_tensor` ([dp_attention.py:587](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L587)) | `_reduce_scatter_tensor` ([ps:753](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L753)) | `ncclReduceScatter` ([pynccl.py:289](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/device_communicators/pynccl.py#L289)) | `communicator_dsa_cp.py:75`（CP MLP 后） |
| **All-to-All** | `DeepEPDispatcher.dispatch/combine` ([deepep.py:897](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L897)/[:908](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L908)) | —（非 GroupCoordinator） | DeepEP `buffer.dispatch/combine`（RDMA+NVLink 自研） | `layer.py:1095`/`:1106`（EP MoE） |

> `ps` = `parallel_state.py`。前三原语经 `GroupCoordinator` 统一分发到 pynccl/aiter/custom；A2A 完全独立，由 `DeepEPBuffer` 自管拓扑感知 kernel。本文 TP 路径用前三个，EP 路径用 A2A 替代前三个（KV AllGather 两者共用）。

---

## 5. TP 模式 MoE（baseline 路径）

`--tp-size 8`，不传 `--ep-size`（`ep_size=1`）。每个 expert 的权重沿 intermediate 维度切 8 份，8 张卡各持 1/8 权重；每张卡都持有全部 256 个 expert，但每个 expert 只有 1/8 的权重。

### 5.1 TP 模式每层 MoE 流程

```
所有 token 在本地 → 走本地 1/8 权重算 partial result
                    │
                    ▼
              all_reduce（8 卡求和）  ← 每层一次，critical path
                    │
                    ▼
              还原完整 hidden_states
```

### 5.2 为什么 ep_size=1 时是 TP 切分

[`parallel_state.py`](../python/sglang/srt/distributed/parallel_state.py) 中的切分计算：

```python
moe_ep_size = expert_model_parallel_size    # = ep_size = 1（baseline）
moe_dp_size = moe_data_model_parallel_size  # = 1（默认）
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
            # = tp_size / 1 / 1 = tp_size = 8
# 结论：ep_size=1 时 moe_tp_size = tp_size
# 每个 expert 的权重按 TP 切分在所有 8 张卡上
```

所以 baseline 每个 rank 虽持有全部 expert，但每个 expert 只持有 1/8 权重（$w_1, w_2, w_3$ 在 intermediate_size 维度切分），计算出的是 **TP partial result**，需要 all-reduce 合并。

### 5.3 TP 模式的瓶颈

| 问题 | 说明 |
|------|------|
| **每层一次 all-reduce** | 数据量 = `hidden_size × num_tokens`，8 卡全同步，卡在 critical path |
| **GPU 利用率差** | 每 expert 被切成 8 份，每卡算瘦 GEMM `[tokens, intermediate/8]`，SM 吃不满 |
| **无法 overlap** | all-reduce 必须等所有 expert partial GEMM 完才能 reduce，纯串行 |
| **长上下文放大** | prefill token 多 → all-reduce 数据爆炸 → comm bound（与本次测试场景完全吻合） |

---

## 6. EP 原理与 DeepEP 实现

`--ep-size 8 --moe-a2a-backend deepep`。换切法：不切 expert 权重，切 expert 本身。256 expert / 8 = 32 expert 每卡，每个 expert 持有**完整权重**。

### 6.1 EP 模式每层 MoE 流程

```
token 看 topk_ids → 该 expert 在哪卡？
        │
        ▼
   all-to-all dispatch（把 token 发到对应 expert 所在的卡）
        │
        ▼
   目标卡用完整权重算 expert GEMM
        │
        ▼
   all-to-all combine（把结果发回原卡）
        │
        ▼
   按 topk_weights 加权求和 → 还原 hidden_states
```

两次 all-to-all（dispatch 去 + combine 回）替代 TP 的 all-reduce。

### 6.2 DeepEP 模式解析

[`DeepEPMode`](../python/sglang/srt/layers/moe/utils.py)（[utils.py:139](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L139)）三种模式：

```python
class DeepEPMode(Enum):
    NORMAL = "normal"          # 大 batch / prefill
    LOW_LATENCY = "low_latency"  # 小 batch / decode
    AUTO = "auto"              # 按是否 extend_in_batch 自动切换

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self
        if is_extend_in_batch:
            return DeepEPMode.NORMAL      # prefill → normal
        else:
            return DeepEPMode.LOW_LATENCY  # decode → low_latency
```

`--deepep-mode auto` → prefill 走 normal，decode 走 low_latency。两种模式在 `DeepEPDispatcher.__init__`（[deepep.py:818](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L818)）中都会预创建对应 impl：

```python
if self.deepep_mode.enable_low_latency():
    self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(...)
if self.deepep_mode.enable_normal():
    self._normal_dispatcher = _DeepEPDispatcherImplNormal(...)
```

### 6.3 Normal 模式实现（prefill 路径）

[`_DeepEPDispatcherImplNormal`](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py)（[deepep.py:453](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L453)）。核心是两段式 dispatch / combine。**注意：拆开是为跨层 two-batch overlap 服务，单层内仍串行**（`dispatch_b` 后 `current_stream_wait`，[deepep.py:488](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L488)）。

**dispatch**（`_dispatch_core`，[deepep.py:503](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L503)）：

```python
def _dispatch_core(self, x, topk_ids, topk_weights, previous_event):
    buffer = self._get_buffer()
    # 1. 算每卡要发/收多少 token（拓扑感知 layout）
    (num_tokens_per_rank,
     num_tokens_per_rdma_rank,    # 走 RDMA 的跨机 token 数
     num_tokens_per_expert,
     is_token_in_rank,
     previous_event) = buffer.get_dispatch_layout(
        topk_ids, self.num_experts,
        previous_event=previous_event,
        async_finish=self.async_finish,
        allocate_on_comm_stream=previous_event is not None,
    )
    # 2. 真正 all-to-all dispatch
    (recv_x, recv_topk_ids, recv_topk_weights,
     num_recv_tokens_per_expert,
     self.handle,    # combine 时反向用
     event) = buffer.dispatch(
        x, topk_idx=topk_ids, topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=self.async_finish,
        expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
        config=DeepEPConfig.get_instance().normal_dispatch_config,
    )
    return recv_x, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert, event
```

**combine**（`_combine_core`，[deepep.py:587](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L587)）：用 `self.handle` 反向 all-to-all。

```python
def _combine_core(self, x, previous_event):
    buffer = self._get_buffer()
    combined_x, _, event = buffer.combine(
        x, self.handle,
        async_finish=self.async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None,
        config=DeepEPConfig.get_instance().normal_combine_config,
    )
    return combined_x, event
```

**关键设计**：
- DeepEP buffer 自己实现 **RDMA + NVLink 拓扑感知 kernel**（`num_tokens_per_rdma_rank` 走 RDMA 跨机，其余走 NVLink 同机），不是 `torch.distributed.all_to_all`。
- `async_finish=True` + `previous_event=Buffer.capture()` 让通信落在 comm stream。**但 `dispatch_b`/`combine_b` 末尾 `event.current_stream_wait()` 把 compute stream 拉回等待**（[deepep.py:488/582](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L488)），单层内 dispatch→GEMM→combine 仍串行，comm/compute 不 overlap。
- `dispatch_a` / `dispatch_b` 拆开（[deepep.py:885-901](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L885-L901)）：`dispatch_a` 做 layout + capture，`dispatch_b` 做真正通信，配合 [two_batch_overlap.py](../python/sglang/srt/batch_overlap/two_batch_overlap.py) 与 `overlap_args` 实现**跨层**通信/计算重叠（TBO 需显式开启）。

### 6.4 Low-latency 模式实现（decode 路径）

[`_DeepEPDispatcherImplLowLatency`](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py)（[deepep.py:613](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L613)），专为低 batch 解码延迟优化：

- `low_latency_dispatch` / `low_latency_combine` 使用固定 buffer，避免动态 layout 开销
- `return_recv_hook` 机制：用 hook 替代 `event.current_stream_wait()`，更激进 overlap
- 文档约束：**decode batch 应小于 256**（[deepep.py:618](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L618)，源自 DeepEP 官方建议）

```python
# deepep.py:617-620
"""
num_max_dispatch_tokens_per_rank: the actual batch size in the decoding
engine should be less than 256
https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
"""
```

combine 阶段还支持与计算的细粒度 overlap（`combine_b`，[deepep.py:738](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L738)），通过 `overlap_args.stream` 在独立 stream 上跑 combine，Blackwell 与非 Blackwell 走不同 signal 机制。

### 6.5 EP 模式下的切分计算

`ep_size = tp_size = 8` 时：

```python
moe_ep_size = 8
moe_tp_size = tp_size // moe_ep_size // moe_dp_size
            # = 8 / 8 / 1 = 1
# 结论：ep_size = tp_size 时 moe_tp_size = 1
# 每个 expert 的权重不再做 TP 切分
# 每个 rank 持有 E/P = 256/8 = 32 个 expert 的完整权重
```

---

## 7. CP=8 下 TP 与 EP 的并行度与通信对比

当 `--attn-cp-size 8` 启用 DSA Prefill CP 时，round-robin-split 自动设置 `enable_dp_attention = True`、`attn_cp_size = tp_size` 和 `moe_dense_tp_size = 1`（[server_args.py:1871-1882](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1871-L1882)）。

关键公式（[parallel_state.py:1931](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L1931) / [parallel_state.py:2003](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L2003)）：
- `attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size`
- `moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size`

CP=8, TP=8 时：`attn_tp_size = 8 // 8 // 1 = 1`——attention 权重不做 TP 切分，每卡持有完整权重，各自独立计算自己序列片段的 attention。**TP=8 与 EP=8 在 attention 侧完全相同，差异仅来自 MLP/MoE 部分。**

> 注意：`moe_dense_tp_size=1` 影响 dense MLP 层不做 TP 切分（使 `enable_moe_dense_fully_dp()=True`），但 **Attention TP=1 的根本原因是 `attn_cp_size=8` 把 TP 退化为 1**（[server_args.py:1879-1881](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1879-L1881) 注释明确说明"Keep attn_tp_size == 1 under DSA CP"）。

### 7.1 并行度分解表

| 组件 | TP=8（CP=8，无 EP） | EP=8（CP=8，DeepEP） |
|------|---------------------|----------------------|
| **Attention 有效 TP** | 1（`attn_tp = tp/cp/dp = 8/8/1`，CP 吞掉 TP） | 1（同左） |
| **Attention 权重** | 完整（attn_tp_size=1，每卡复制） | 完整（同左） |
| **Attention 通信** | KV AllGather（跨 CP rank 收集 KV 分片） | KV AllGather（同左） |
| **MoE 有效 TP** | 8（`moe_tp = tp/ep/dp = 8/1/1`） | 1（`moe_tp = 8/8/1`） |
| **MoE EP** | 1（`ep_size=1`，不切 expert） | 8（`ep_size=8`，按 expert 切） |
| **Expert 分布** | 每卡 256 expert 的 1/8 权重 | 每卡 32 个完整 expert |
| **MoE 通信** | AllReduce（partial result 合并，[deepseek_v2.py:1034](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1034)） | A2A dispatch + combine |
| **CP MLP 前通信** | AllGather（`DSACPLayerCommunicator`，[communicator_dsa_cp.py:185](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L185)） | —（跳过，A2A 替代） |
| **CP MLP 后通信** | ReduceScatter（`DSACPLayerCommunicator`，[communicator_dsa_cp.py:230](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L230)） | —（跳过，A2A 替代） |
| **每层同步通信总计** | KV AllGather + AllGather + AllReduce + ReduceScatter = **4 次** | KV AllGather = **1 次** + 2 次 pairwise A2A |

### 7.2 通信流程图

#### TP=8（CP=8，无 EP）

> 通信由 [`DSACPLayerCommunicator`](../python/sglang/srt/layers/communicator_dsa_cp.py) 控制。
> 注意：`attn_tp_size=1` 导致 `FULL` 和 `TP_ATTN_FULL` 分组大小相同（=1），
> 但 DSA CP 通信器根据 scatter mode 枚举值（`FULL` ≠ `SCATTERED`）仍然执行 gather/scatter。

```
CP Rank 0          CP Rank 1       ···       CP Rank 7
(序列 1/8)          (序列 1/8)                (序列 1/8)
    │                   │                         │
    ▼                   ▼                         ▼
┌────────────── KV AllGather ──────────────────────────┐
│  各卡收集全部 CP rank 的 KV 分片 → 拼成完整 KV       │
└──────────────────────────────────────────────────────┘
    │                   │                         │
    ▼                   ▼                         ▼
┌──────────┐      ┌──────────┐               ┌──────────┐
│Attention │      │Attention │               │Attention │
│DP,TP=1   │      │DP,TP=1   │               │DP,TP=1   │
└────┬─────┘      └────┬─────┘               └────┬─────┘
     │                 │                          │
     ▼                 ▼                          ▼
┌──────────── AllGather (SCATTERED→FULL) ─────────────┐
│  dsa_cp_gather_hidden_states (communicator_dsa_cp:185)│
│  跨 CP rank 收集 hidden_states → 每卡拿到完整 token  │
└──────────────────────────────────────────────────────┘
     │                 │                          │
     ▼                 ▼                          ▼
┌──────────────── AllReduce ──────────────────────────┐
│  deepseek_v2.py:1034                                │
│  每卡用 1/8 权重算 partial → 8 卡求和还原           │
│  ⚠️ 全员同步，必须等所有卡 partial GEMM 完才能reduce │
└──────────────────────────────────────────────────────┘
     │                 │                          │
     ▼                 ▼                          ▼
┌──────── ReduceScatter (FULL→SCATTERED) ────────────┐
│  dsa_cp_reduce_scatter_hidden_states (dsa_cp:230)   │
│  MoE 输出按 CP rank 切分，每卡只保留自己的 1/8      │
└──────────────────────────────────────────────────────┘
     │                 │                          │
     ▼                 ▼                          ▼
  下一层             下一层                    下一层

◆ 每层 4 次同步通信：KV AllGather + AllGather + AllReduce + ReduceScatter
◆ AllReduce 必须等所有卡完成 → 串行，但 MoE 内部仅 1 次集合通信，NVLink 带宽利用率高
◆ 稳态大 batch prefill 下实际快于 EP（[第 2.1 节](#21-吞吐与-ttft-对比)）
```

#### EP=8（CP=8，DeepEP）

> A2A 模式下 `mlp_mode=SCATTERED`（[communicator.py:382-385](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator.py#L382-L385)），
> `DSACPLayerCommunicator` 走 `_simple`（无 CP gather/scatter），
> token 分发/回收由 DeepEP A2A 处理。

```
CP Rank 0          CP Rank 1       ···       CP Rank 7
(序列 1/8)          (序列 1/8)                (序列 1/8)
    │                   │                         │
    ▼                   ▼                         ▼
┌────────────── KV AllGather ──────────────────────────┐
│  各卡收集全部 CP rank 的 KV 分片 → 拼成完整 KV       │
└──────────────────────────────────────────────────────┘
    │                   │                         │
    ▼                   ▼                         ▼
┌──────────┐      ┌──────────┐               ┌──────────┐
│Attention │      │Attention │               │Attention │
│DP,TP=1   │      │DP,TP=1   │               │DP,TP=1   │
└────┬─────┘      └────┬─────┘               └────┬─────┘
     │                 │                          │
     ▼                 ▼                          ▼
┌──────────── A2A dispatch ──────────────────────────┐
│  按 topk_ids 将 token 定向发到 expert 所在卡        │
│  ✅ pairwise 通信，数据驱动                          │
│  ⚠️ normal 模式单层内串行（dispatch 后 wait）        │
│  ✅ 无需先 AllGather 收集全卡 token 即可定向分发    │
└────────────────────────────────────────────────────┘
     │                 │                          │
     ▼                 ▼                          ▼
┌──────────┐      ┌──────────┐               ┌──────────┐
│MoE Expert│      │MoE Expert│               │MoE Expert│
│TP=1      │      │TP=1      │               │TP=1      │
│32 完整   │      │32 完整   │               │32 完整   │
│expert    │      │expert    │               │expert    │
└────┬─────┘      └────┬─────┘               └────┬─────┘
     │                 │                          │
     ▼                 ▼                          ▼
┌───────────── A2A combine ─────────────────────────┐
│  expert 结果定向发回 token 原始所在卡               │
│  → 按 topk_weights 加权求和还原 hidden_states      │
│  ⚠️ pairwise 通信，单层内串行                        │
│  ✅ 替代 TP 的 AllReduce + ReduceScatter            │
└────────────────────────────────────────────────────┘
     │                 │                          │
     ▼                 ▼                          ▼
  下一层             下一层                    下一层

◆ 每层 1 次同步 + 2 次 pairwise：KV AllGather + A2A ×2
◆ ⚠️ normal 模式单层内 A2A 串行无 overlap（dispatch 后 `current_stream_wait`），仅跨层 TBO 可 overlap
◆ forward_deepep 无 AllReduce（moe_tp_size=1，deepseek_v2.py:1097-1309）
```

### 7.3 通信关键差异

| | TP=8 | EP=8（DeepEP） |
|---|---|---|
| **同步通信/层** | 4 次（KV AllGather + AllGather + AllReduce + ReduceScatter） | 1 次（KV AllGather） + 2 次 pairwise A2A |
| **可 overlap** | ❌ AllReduce 必须等所有 partial 完成 | ❌ **normal 模式单层内串行**（dispatch 后 `current_stream_wait`），仅跨层 TBO 可 overlap |
| **数据感知** | ❌ 通信 pattern 固定，与 token 路由无关 | ✅ 通信由 topk_ids 决定，稀疏激活时更高效 |
| **通信瓶颈** | AllReduce 量 = hidden_size × 全部 token | A2A 量 = 仅被路由的 token |
| **CP MLP 通信** | 需要 AllGather + ReduceScatter（`DSACPLayerCommunicator`） | 无需（A2A 天然完成分发与回收，`mlp_mode=SCATTERED`） |
| **MoE 内部 AllReduce** | 需要（`moe_tp_size=8 > 1`，[deepseek_v2.py:1034](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1034)） | 不需要（`moe_tp_size=1`，`forward_deepep` 无 AllReduce） |

---

## 8. 为什么稳态大 batch prefill 下 in-seq-split 反而慢于 round-robin-split

> 即 in-seq-split（EP+DeepEP）慢于 round-robin-split（TP）。下表保留 TP vs EP 理论维度对比，但**理论优势在大 batch prefill 稳态下被通信劣势吞掉**（详见 [8.1](#81-针对-dsa-prefill-cp-场景稳态下-ep-反而慢的两个原因)）。

| 维度 | TP=8（baseline） | EP=8+DeepEP（对照） |
|------|------------------|---------------------|
| expert 权重分布 | 切 8 份，每卡全 256 expert 的 1/8 | 每卡 32 个完整 expert |
| 每层通信 | all-reduce（全同步 ring/tree） | 2× all-to-all（pairwise 定向） |
| 通信 kernel | NCCL ring/tree | DeepEP RDMA+NVLink 拓扑感知 |
| overlap 能力 | 难，reduce 必须等 partial 完成 | **单层内无 overlap**（normal 模式 `dispatch_b` 后 `current_stream_wait`，[deepep.py:488](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L488)）；`dispatch_a/b` 拆分仅服务跨层 TBO |
| GEMM 效率 | 每 expert 瘦 GEMM ×8 卡，SM 吃不满 | 完整 expert 胖 GEMM，单卡算满，tensorcore 利用率高 |
| 显存 | expert 权重 TP 切片（无冗余） | expert 权重 EP 分担（无冗余） |

### 8.1 针对 DSA Prefill CP 场景：稳态下 EP 反而慢的两个原因

本次稳态测试：64 并发 × 22934 token ≈ 1.47M token 大 batch prefill，TBO 未开。此场景下 EP 的通信劣势暴露：

**① DeepEP normal 模式单层内无 overlap，A2A 全在 critical path**
[`FusedMoE.forward_impl`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1091-L1116) 是纯串行 `dispatch → run_moe_core → combine`。`async_finish=True` 只让通信落 comm stream，但 [`dispatch_b` 后立即 `event.current_stream_wait()`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L488)（combine 同理 [L582](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L582)），compute stream 必须等 dispatch 完才能跑 expert GEMM。`dispatch_a/b` 拆分是为 **two-batch overlap**（跨层），单层内无 overlap。TBO 未开 → 两次 A2A 全程串行卡在 critical path。
TP 路径 MoE 内部虽也有 1 次串行 AllReduce，但 NVLink 上 AllReduce（内部 RS+AG）带宽利用率高；EP 则是 2 次 pairwise A2A 全程串行。注意此处只比 MoE 内部通信，TP 的 CP 侧 AllGather/ReduceScatter 另算（见 [第 8.2 节](#82-cp-模式下的通信次数对比) 全层 4 次 vs 1 次+2 A2A）。

**② A2A 通信量被 top_k 放大**
DeepEP 模式 [`_is_deepep_fusion`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L543-L551) 把 shared expert 融进 MoE：top_k 8→9，expert 数 256→264（`n_routed_experts + moe_ep_size`，EP=8 时 256+8）。每 token 按 9 份发往 9 个 expert，dispatch + combine 两次 A2A 共 ~9× 放大。A2A 总量在大 batch 下压过 TP 单次 AllReduce。

> 注：[`params_bytes=2`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L332) 仅用于 **buffer 预分配 sizing**（`hidden_bytes = hidden_size * param_bytes`，[deepep.py:176](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L176)），非实际通信 dtype。实际通信 dtype 由 [`deepep_output_dtype`](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L186) 决定，DeepSeek-V3 FP8 量化下默认 **FP8 通信**（`use_fp8=True`，[deepep.py:387](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L387)），非 bf16。故「bf16 通信翻倍」说法不成立——旧版第 4 点已删。

**③ shared expert fusion 对两条路径的影响**
默认 `disable_shared_experts_fusion=False`（[server_args.py:783](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L783)）→ `num_fused_shared_experts = n_shared_experts > 0`（[deepseek_v2.py:538](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L538)），即 **shared expert 默认就融进 MoE kernel**，TP 与 EP 两条路径都如此：
- TP 路径：`forward_normal_dual_stream` 的入口条件含 `num_fused_shared_experts == 0`（[deepseek_v2.py:831](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L831)），默认 fusion-on 下**不满足**，走 `forward_normal`（[L850](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L850)），shared 已在 MoE kernel 内，无独立 shared stream。
- EP 路径：`_is_deepep_fusion=True`，shared 作为本地 expert slot 融进 MoE（[deepseek_v2.py:543-551](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L543-L551)），`forward_deepep` 中 `num_fused_shared_experts == 0` 分支（[L1115](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1115)）不进入，同样无独立 shared stream（除非开 SBO，`_fuse_shared_experts_inside_sbo`）。

> ⚠️ **修正旧版错误**：旧版称「TP 用 `forward_normal_dual_stream` 把 shared 放 alt_stream，EP 失去该 overlap」——此说法在默认 fusion-on 配置下**不成立**：两条路径都把 shared 融进 MoE kernel，都没有独立 shared dual-stream。该差异仅在 `--disable-shared-experts-fusion`（fusion-off）时才出现，且此时 TP 走 `forward_normal_dual_stream`、EP 走 `forward_deepep` 的 `num_fused_shared_experts==0` 分支（[L1115-L1121](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1115-L1121)），**两条路径仍都有 alt_stream shared overlap**，无 EP 专属劣势。故 fusion/dual-stream 不是本次 EP 慢的根因，删除旧版第 3 点。本次 EP 慢的主因是 ①（A2A 串行）+ ②（top_k 放大 A2A 量）。

> **GEMM 效率维度 EP 仍占优**：TP 每 expert 瘦 GEMM（`[tokens, intermediate/8]`），EP 完整 expert 胖 GEMM。但大 batch prefill 下 GEMM 不是瓶颈，通信才是，故 GEMM 效率优势被通信劣势吞掉。

### 8.2 CP 模式下的通信次数对比

在 DSA Prefill CP 下，每层集合通信次数（TP 行 = round-robin-split，a2a 行 = in-seq-split；详见 [`dsa_prefill_cp_round_robin_math.md`](dsa_prefill_cp_round_robin_math.md) 第 5 节）：

| 模式 | KV AllGather | MLP 前 AllGather | MoE AllReduce | MLP 后 ReduceScatter | 每层集合通信 |
|------|:---:|:---:|:---:|:---:|:---:|
| 非 a2a（TP baseline） | 1 次 | 1 次 | 1 次 | 1 次 | **4 次** |
| a2a（DeepEP） | 1 次 | 0 次 | 0 次 | 0 次 | **1 次** + 2 A2A |

a2a 模式下 MLP 前 AllGather、MoE 内部 AllReduce 与 MLP 后 ReduceScatter 均被跳过（由 DeepEP all-to-all dispatch + combine 替代），每层同步通信从 4 次降到 1 次。

> 注意：DeepEP 的 all-to-all 本身也是通信，替代了 TP 路径里 MoE 内部的 post-experts all-reduce + CP 的 AllGather/ReduceScatter 三次同步操作。理论上定向、可 overlap，但 normal 模式实现单层内串行（见 [第 8.1 节](#81-针对-dsa-prefill-cp-场景稳态下-ep-反而慢的两个原因)）。

---

## 9. 代码路径速查

| 概念 | 文件 / 位置 | 说明 |
|------|------------|------|
| `DeepEPDispatcher` | [deepep.py:818](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L818) | DeepEP 顶层 dispatcher，按 mode 分发到 normal/low-latency impl |
| `_DeepEPDispatcherImplNormal` | [deepep.py:453](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L453) | prefill / 大 batch 路径，两段式 dispatch/combine |
| `_DeepEPDispatcherImplLowLatency` | [deepep.py:613](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L613) | decode / 小 batch 路径，固定 buffer + return_recv_hook |
| `DeepEPMode` | [utils.py:139](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L139) | 枚举 NORMAL / LOW_LATENCY / AUTO，`resolve()` 按 extend_in_batch 切换 |
| `_dispatch_core`（normal） | [deepep.py:503](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L503) | layout 计算 + `buffer.dispatch` 真正 all-to-all |
| `_combine_core`（normal） | [deepep.py:587](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L587) | `buffer.combine` 反向 all-to-all |
| `DeepEPBuffer` | [deepep.py:150](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L150) | RDMA+NVLink 拓扑感知通信 buffer |
| `DeepEPMoE` | [ep_moe/layer.py:47](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/ep_moe/layer.py#L47) | EP MoE 层，继承 `FusedMoE` |
| `moe_tp_size` 计算 | `parallel_state.py` | `tp_size // ep_size // dp_size`，决定 expert 权重是否 TP 切分 |
| `ep_size` 默认值 | [server_args.py:635](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L635) | 默认 `1`（不开 EP） |
| `moe_a2a_backend` 默认值 | [server_args.py:636](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L636) | 默认 `"none"`（不开 DeepEP） |
| round-robin-split 自动配置 | [server_args.py:1870-1875](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1870-L1875) | 只设 `enable_dp_attention` / `moe_dense_tp_size`，**不自动开 DeepEP/EP** |
| in-seq-split 自动配置 | [server_args.py:1861-1868](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1861-L1868) | 自动强制 `moe_a2a_backend=deepep` + `ep_size=tp_size` |

---

## 10. 结论与建议

1. **稳态大 batch prefill 下 EP+DeepEP 慢于 TP-only MoE**（[第 2.1 节](#21-吞吐与-ttft-对比)）：72k vs 89k tok/s（−19%），TTFT +25%。根因是 DeepEP normal 模式单层内 A2A 串行无 overlap + top_k 放大通信量（[第 8.1 节](#81-针对-dsa-prefill-cp-场景稳态下-ep-反而慢的两个原因)）。
2. **EP 的优势区在小 batch / decode / 长 context 单序列 / TBO 开启**时（[第 2.3 节](#23-何时-ep-会赢)）。混合 prefill+decode 负载仍推荐 `--deepep-mode auto`：prefill normal，decode low-latency（bs<256）。
3. **round-robin-split 模式默认不开 EP**。与 in-seq-split（自动强制 DeepEP）不同，round-robin-split 需手动传 `--moe-a2a-backend deepep --ep-size <tp_size>` 才能切到 EP 路径。纯 prefill 重载若以 round-robin-split 为主，稳态下 TP 路径更快。
4. **变量控制**：基准测试建议固定 `--cuda-graph-max-bs`、关闭/开启 radix cache 等参数一致，且**预热后再取数**（冷启动方差会误导稳态结论）。
5. **进一步验证**：可用 nsys profile 对比 TP AllReduce 与 DeepEP A2A 的实际时间占比，定位剩余瓶颈。参见 [`llm-torch-profiler-analysis`](../.claude/skills/llm-torch-profiler-analysis) skill 与 [`generate-profile`](../.claude/skills/generate-profile) skill。

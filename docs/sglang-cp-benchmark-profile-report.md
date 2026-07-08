# SGLang CP Benchmark & Profile 综合报告

> 模型: GLM-5.1-FP8 (10层稀疏MoE) | GPU: 8×H20 (95GB) | 测试: 64个 sonnet-10k prompt, output_len=1
> 采集时间: 2026-06 ~ 2026-07

---

## 1. 测试环境

| 项目 | 详情 |
|------|------|
| 模型 | GLM-5.1-FP8-10L (10层稀疏MoE, GlmMoeDsaForCausalLM) |
| GPU | 8×NVIDIA H20 (95GB each) |
| 容器 | `sglang-debug` (uhub.service.ucloud.cn/umirror/sglang:v0.5.12.post1-cu130) |
| sglang 版本 | v0.5.12.post1 |
| 代码挂载 | `/upfs/swan/modsglang/srt` → 容器内 `/sgl-workspace/sglang/python/sglang/srt` |
| 测试工具 | `vllm bench serve` (v0.16.1rc1) |
| 数据集 | sonnet.txt (~22,934 tokens), 64 prompts, output_tokens=1, max_concurrency=64 |
| Tokenizer | `/upfs/swan/models/GLM-5.1-FP8/GLM-5.1-FP8-10L` |

---

## 2. 三种配置说明

| 配置 | sgland 启动参数 | MoE token 分发方式 | 核心特点 |
|------|----------------|-------------------|---------|
| **No-CP** | `--tp-size 8` (无 CP 相关参数) | TP Ring AllGather → 每卡全量 token → 本地专家 | 最简单，无 CP 通信，但每卡处理全序列 |
| **CP Round-Robin** | `--tp-size 8 --enable-nsa-prefill-context-parallel --attn-cp-size 8 --nsa-prefill-cp-mode round-robin-split` | 本地 round-robin 分配 → 各卡独立处理本地专家 | 无额外 MoE 通信，attention 按 CP 切分 |
| **CP+DeepEP in-seq** | `--tp-size 8 --enable-nsa-prefill-context-parallel --attn-cp-size 8 --nsa-prefill-cp-mode in-seq-split --moe-a2a-backend deepep --ep-size 8 --deepep-mode auto --cuda-graph-max-bs 128` | DeepEP dispatch → 跨卡 all-to-all 分发到专家所在卡 | Token 按专家归属跨卡分发，需额外通信开销 |

**三种配置的数据流对比：**

```
No-CP (纯 TP=8):
  Request → TP Ring AllGather → 每卡看全序列 → 本地 MoE → TP AllReduce
  ❌ 每卡需处理全序列，序列长时计算量大
  ❌ 8卡 TP Ring 串行7跳，AllReduce 开销大

CP Round-Robin (TP=1, CP=8):
  Request → CP AllGather (attention) → 各卡 1/8 序列 → 本地 round-robin MoE → CP AllReduce
  ✅ 每卡只处理 1/8 序列，计算量降低 8x
  ✅ MoE 无跨卡通信（本地 round-robin）
  ❌ MoE 负载可能不均衡（某些卡热门专家多）

CP+DeepEP in-seq (TP=1, CP=8, EP=8):
  Request → CP AllGather (attention) → 各卡 1/8 序列 → DeepEP dispatch (跨卡) → 专家计算 → DeepEP combine (跨卡)
  ✅ 每卡只处理 1/8 序列
  ✅ Token 精确路由到专家所在卡，负载均衡
  ❌ DeepEP dispatch/combine 额外通信开销 (8.57%)
  ❌ 同步等待 (notify) 开销 > 数据传输开销
```

---

## 3. Benchmark 吞吐量对比

### 3.1 关闭 Radix Cache (--disable-radix-cache)

| 配置 | 同 prompt tok/s | 同 prompt TTFT | 不同 prompt tok/s | 不同 prompt TTFT | vs No-CP |
|------|----------------|----------------|-------------------|------------------|----------|
| **No-CP** (纯 TP=8) | 17,965 | 70,538ms | 26,735 | 27,981ms | 基线 |
| **CP Round-Robin** | 88,880 | 8,493ms | 88,313 | 8,658ms | **5.0x** |
| **CP+DeepEP in-seq** | 53,669 | 17,211ms | 67,305 | 11,716ms | **3.0x** |
| **CP+DeepEP in-seq + SBO** | 72,645 | — | 72,730 | — | **4.0x** |

### 3.2 开启 Radix Cache (默认)

| 配置 | Best tok/s | Best TTFT | 说明 |
|------|-----------|-----------|------|
| **No-CP** | ~89,505 | — | 缓存命中后大幅加速 |
| **CP Round-Robin** | ~467,227 | 2,157ms | 缓存 + CP 双重加速 |
| **CP Round-Robin + DeepEP** | ~464,597 | 2,185ms | 缓存命中时 DeepEP 无收益 (-0.6%) |

> **关键发现**: Radix cache 开启时，DeepEP 几乎无收益 (467k vs 465k, -0.6%)，因为缓存命中后 MoE 计算量极少，DeepEP 的通信开销无法被摊薄。Radix cache 是最大的性能因子 (5x 差距)。

### 3.3 吞吐量可视化

```
吞吐量对比 (--disable-radix-cache, 同 prompt):

No-CP              ████████████████████  17,965 tok/s
CP+DeepEP in-seq   ████████████████████████████████████████████████████████  53,669 tok/s
CP+DeepEP+SBO      ████████████████████████████████████████████████████████████████████████████  72,645 tok/s
CP Round-Robin     ████████████████████████████████████████████████████████████████████████████████████████████████  88,880 tok/s
                   ├──────────────────────────────────────────┤
                   0                                       100,000 tok/s
```

---

## 4. Nsight Systems Profile 分析

### 4.1 Profile 概览

| 指标 | CP Round-Robin | CP+DeepEP in-seq |
|------|---------------|-----------------|
| nsys-rep 文件 | `cp_rr_nocache.nsys-rep` (540M) | `cp_deepep_inseq_nocache.nsys-rep` (540M) |
| 总 GPU Kernel 数 | 1,033,240 | 1,539,984 |
| **总 GPU 时间（8卡累计）** | **138.62s** | **147.39s** |
| Wall Clock 时间 | 19.01s | 24.03s |
| 每卡平均 GPU 时间 | 17.33s | 18.42s |

> No-CP 的 nsys profile (`nocp_nocache.nsys-rep`, 608M) 已采集，但未做详细 kernel 级分析。根据 benchmark 数据推断，No-CP 下每卡需处理全序列，attention 计算是 CP 模式的 ~8x，是主要瓶颈。

### 4.2 GPU Kernel 时间分布（大类对比）

| 类别 | CP Round-Robin | CP+DeepEP in-seq | 差异 |
|------|---------------|-----------------|------|
| **MoE 计算** | **50.6%** (70.19s) | **44.9%** (66.15s) | DeepEP 用 grouped GEMM 更高效 |
| **Attention** | **29.1%** (40.36s) | **27.9%** (41.12s) | 绝对时间接近 |
| **NCCL 通信** | **12.9%** (17.82s) | **12.8%** (18.90s) | 持平 |
| **DeepEP 通信** | — | **8.6%** (12.64s) | DeepEP 独有开销 |
| 其他 + 量化 + 激活等 | 7.5% | 5.9% | — |

```
分类占比对比:

                    CP Round-Robin                CP+DeepEP in-seq
MoE 计算            ████████████████████  50.6%   ████████████████  44.9%
Attention           ████████████ 29.1%            ███████████ 27.9%
NCCL 通信           █████ 12.9%                    █████ 12.8%
DeepEP 通信         —                              ███ 8.6%
Other+Quant+...     ██ 7.5%                        ██ 5.9%
─────────────────────────────────────────────────────────────────────
通信合计            12.9%                          21.4%  ← DeepEP 额外 +8.5pp
```

### 4.3 通信详细分解

#### NCCL 通信

| 通信类型 | CP Round-Robin | CP+DeepEP in-seq | 说明 |
|---------|---------------|-----------------|------|
| **AllGather** | 8.91% (12.36s) | 11.30% (16.66s) | DeepEP 更多（EP 也需 AG） |
| **ReduceScatter** | 3.27% (4.53s) | — | RR 独有（TP o_proj reduce） |
| **AllReduce** | 0.67% (0.93s) | 1.52% (2.24s) | DeepEP 模式下 AR 慢 2.4x |
| **NCCL 小计** | **12.85%** (17.82s) | **12.82%** (18.90s) | — |

**AllGather 延迟分布关键差异：**

| 延迟范围 | RR 调用占比 | DeepEP 调用占比 | 说明 |
|----------|-----------|---------------|------|
| <0.1ms | **71.0%** | 51.8% | RR 大部分 AG 很快完成 |
| 0.1-0.5ms | 28.7% | 31.5% | — |
| 0.5-5ms | 0.2% | **16.5%** | DeepEP 更多中间延迟 |
| 50-500ms | 17.4%(时间占比) | **29.4%**(时间占比) | DeepEP 尾部延迟严重 |

> DeepEP 的 AllGather 延迟分布更差 — 仅 51.8% 调用在 <0.1ms 完成（vs RR 的 71%），原因是 DeepEP 与 NCCL 共享 NVLink 带宽导致竞争。

#### DeepEP 通信（仅 CP+DeepEP in-seq）

| DeepEP 操作 | 占比 | 总时间 | 调用数 | 平均延迟 | 说明 |
|-------------|------|-------|-------|---------|------|
| **cached_notify_combine** | 3.19% | 4,705ms | 10,752 | 0.438ms | 等待对端同步，最大瓶颈 |
| **combine** | 1.88% | 2,765ms | 10,752 | 0.257ms | Token 回收 |
| **notify_dispatch** | 1.58% | 2,332ms | 10,752 | 0.217ms | Dispatch 通知 |
| **dispatch** | 1.04% | 1,530ms | 10,752 | 0.142ms | Token 分发 |
| ep_gather/scatter/layout | 0.89% | 1,304ms | — | — | 辅助操作 |
| **DeepEP 合计** | **8.57%** | **12,638ms** | **86,024** | — | — |

> **核心发现**: 同步开销 (notify_dispatch + cached_notify_combine = 4.77%) **超过** 数据传输开销 (dispatch + combine = 2.92%)。DeepEP 的瓶颈不在数据搬运，而在多卡同步等待。

### 4.4 计算详细分解

#### Attention 子类别

| Attention 子操作 | CP Round-Robin | CP+DeepEP in-seq |
|-----------------|---------------|-----------------|
| **sparse_attn** (NSA) | 23.84% (33.04s) | 22.43% (33.06s) |
| **MQA logits** | 4.36% (6.05s) | 4.62% (6.81s) |
| flash_attn/flashinfer | 0.38% | 0.36% |
| concat_mla | 0.48% | 0.45% |
| **Attention 合计** | **29.11%** | **27.90%** |

> sparse_attn 在两种模式下几乎完全相同 (~2.15ms/call)，因为 CP 切分后每卡处理 1/8 序列，attention 计算量相同。

#### MoE 计算子类别

**CP Round-Robin (fused_moe + DeepGEMM):**

| MoE 子操作 | 占比 | 总时间 | 平均延迟 |
|-----------|------|-------|---------|
| fused_moe_kernel (Triton) | 21.66% | 30.02s | 1.396ms |
| sm90_fp8_gemm (共享专家 DeepGEMM) | 22.07% | 30.59s | 0.356ms |
| gatherTopK | 3.31% | 4.58s | 0.142ms |
| moe_sum_reduce_warp | 2.07% | 2.87s | 0.267ms |
| topk_transform | 1.04% | 1.44s | 0.094ms |
| **MoE 合计** | **50.63%** | **70.19s** | — |

**CP+DeepEP (DeepGEMM grouped GEMM):**

| MoE 子操作 | 占比 | 总时间 | 平均延迟 |
|-----------|------|-------|---------|
| sm90_fp8_gemm (DeepGEMM grouped) | 43.32% | 63.84s | 0.495ms |
| topk_transform | 1.00% | 1.47s | 0.048ms |
| grouped_topk | 0.17% | 0.25s | 0.023ms |
| **MoE 合计** | **44.88%** | **66.15s** | — |

> **关键差异**:
> - RR 用 `fused_moe_kernel (Triton)`：每卡独立处理本地专家，大 batch grid (>2048)，1.4ms/call
> - DeepEP 用 `DeepGEMM (FP8 grouped GEMM)`：跨卡分发 token 集中做 grouped GEMM，小 batch 多组 (grid≤128)，0.5ms/call
> - DeepEP 省掉了 gatherTopK (3.3%)、moe_sum (2.1%) 等辅助 kernel，MoE 计算绝对时间更少 (66s vs 70s)
> - **但节省的 4s 完全被 DeepEP 通信 12.6s 抵消**

### 4.5 各卡通信时间分布

**CP Round-Robin — 负载不均衡明显：**

| GPU | AllGather | AllReduce | ReduceScatter | 通信合计 | 通信占比 |
|-----|----------|----------|--------------|---------|---------|
| 0 | 2,273ms | 119ms | 607ms | 2,999ms | 16.6% |
| 1 | 1,471ms | 127ms | 590ms | 2,187ms | 12.7% |
| 2 | 1,676ms | 104ms | 593ms | 2,373ms | 13.6% |
| 3 | 1,324ms | 119ms | 613ms | 2,055ms | 12.0% |
| 4 | 855ms | 108ms | 562ms | 1,524ms | **9.2%** |
| 5 | 2,009ms | 111ms | 573ms | 2,692ms | 15.1% |
| 6 | 873ms | 117ms | 387ms | 1,377ms | **8.4%** |
| 7 | 1,878ms | 129ms | 604ms | 2,611ms | 14.7% |

> RR 模式下 GPU 间通信差异达 2x (1,377ms vs 2,999ms)，原因是 NCCL RING 算法中不同 rank 起始位置不同。

**CP+DeepEP — 更均匀但绝对值更高：**

| GPU | AllGather | AllReduce | DeepEP | 通信合计 | 通信占比 |
|-----|----------|----------|--------|---------|---------|
| 0 | 1,983ms | 282ms | 1,299ms | 3,820ms | 20.8% |
| 3 | 2,136ms | 288ms | 1,579ms | 4,271ms | **23.1%** |
| 6 | 2,114ms | 266ms | 1,242ms | 3,936ms | 21.3% |
| **平均** | **2,087ms** | **279ms** | **1,312ms** | **3,942ms** | **21.4%** |

> DeepEP 模式下通信更均匀 (20.6%-23.1%)，但每卡通信绝对值更高 (3.8-4.3s vs 1.4-3.0s)。

---

## 5. SBO 优化效果

SBO (`--enable-single-batch-overlap`) 通过重叠 DeepEP dispatch/combine 与专家计算，减少通信感知时间。

| 场景 | 同 prompt tok/s | 不同 prompt tok/s | 提升 |
|------|----------------|-------------------|------|
| CP+DeepEP in-seq (无 SBO) | 53,669 | 67,305 | 基线 |
| **CP+DeepEP in-seq + SBO** | **72,645** | **72,730** | **+35% / +8%** |

**SBO 效果分析：**
- 同 prompt: +35% (54k → 73k)，收益最大
- 不同 prompt: +8% (67k → 73k)，收益较小
- SBO 让同/不同 prompt 性能一致化 (~73k)，消除了之前的差异
- 通信占比从 ~20.5% 降至 ~15%

**SBO 原理：**
- `dispatch_shared_one_stream_overlap`: DeepEP dispatch 与 shared expert 计算重叠
- `combine_down_gemm_two_stream_overlap`: DeepEP combine 与 MoE down_proj 重叠
- 利用双 CUDA stream 实现通信-计算流水线

---

## 6. 综合对比与瓶颈分析

### 6.1 三种配置根本差异

| 维度 | No-CP | CP Round-Robin | CP+DeepEP in-seq |
|------|-------|---------------|-----------------|
| 序列处理 | 每卡全序列 | 每卡 1/8 序列 | 每卡 1/8 序列 |
| MoE token 分发 | TP Ring AllGather | 本地 round-robin (无通信) | DeepEP dispatch (跨卡) |
| MoE 计算 | fused_moe (Triton) | fused_moe (Triton) | DeepGEMM (FP8 grouped GEMM) |
| MoE 结果合并 | TP AllReduce | 本地 moe_sum | DeepEP combine (跨卡) |
| 额外通信 | TP AG/AR (固有) | — | +12.64s (8.57%) |
| 吞吐量 | ~18k tok/s | ~89k tok/s | ~54k tok/s (73k w/ SBO) |

### 6.2 各配置瓶颈

**No-CP 瓶颈 — Attention 计算主导：**
- 每卡需处理全序列 (~23k tokens)，attention 计算是 CP 模式的 ~8x
- TP Ring AllReduce 串行 7 跳，O(N) 延迟
- 吞吐仅 18k tok/s，TTFT 达 70s

**CP Round-Robin 瓶颈 — MoE 计算主导 (50.6%)：**
- MoE 计算 (fused_moe + DeepGEMM) 占 43.7%，是最主要的时间消耗
- gatherTopK + moe_sum 等辅助操作占 6.9%
- AllGather 延迟不均 (GPU 间差异 2x)
- 优化空间: 减少辅助开销、优化 fused_moe kernel

**CP+DeepEP in-seq 瓶颈 — 通信额外开销：**
- DeepEP 通信 8.57% 是纯额外开销，同步 (4.77%) > 数据传输 (2.92%)
- AllGather 因 NVLink 竞争变慢 (11.3% vs RR 的 8.9%)
- AllReduce 慢 2.4x (NVLink 带宽被 DeepEP 占用)
- MoE 计算虽然更高效 (66s vs 70s)，但被通信完全抵消

### 6.3 CP Round-Robin vs CP+DeepEP 差距根源

```
Round-Robin:  89k tok/s    计算主导 (81%), 通信 13%
DeepEP+SBO:  73k tok/s    计算主导 (77%), 通信 21% (含 DeepEP 8.6%)
差距: 18%

差距归因:
  DeepEP dispatch/combine:          +5.0s (纯额外)
  DeepEP 同步等待 (notify):         +7.0s (同步 > 传输)
  AllGather 变慢 (NVLink 竞争):     +4.3s (11.3% vs 8.9%)
  AllReduce 变慢 (NVLink 竞争):     +1.3s (2.4x 变慢)
  MoE 计算更高效 (grouped GEMM):    -4.0s (66s vs 70s)
  ─────────────────────────────────────────
  净额外开销:                       ~13.6s → 18% 性能差距
```

---

## 7. 已实施的优化

### 7.1 保留的优化 (communicator.py)

| 优化 | 改动 | 效果 |
|------|------|------|
| 省掉冗余 moe_cp AllGather | `attn_tp_size==1` 时返回 `FULL` | 持平 (冗余 AG 仅 0.18%)，逻辑正确，保留 |
| 放开 AllReduce-LN 融合 | `attn_tp_size!=1` 时才禁止融合 | 对 prefill 无效 (batch > 2048 限制)，保留 |

### 7.2 SBO 重叠优化

| 优化 | 改动 | 效果 |
|------|------|------|
| `--enable-single-batch-overlap` | Dispatch 与 shared expert 重叠，Combine 与 down_proj 重叠 | +35% (同 prompt), +8% (不同 prompt) |

### 7.3 已尝试但失败/回退的优化

| 优化 | 结果 | 原因 |
|------|------|------|
| Prefill dual-stream | 退化 7% (89k→83k) | 大 batch SM 竞争 + stream 同步开销 |
| MoE 后端切换 (cutlass/deep_gemm) | 慢 12% 或崩溃 | triton 已是最优 |
| K/V AllGather 重叠 | 无改善 | NCCL 内部已优化或 NVLink 带宽饱和 |

---

## 8. 下一步优化建议

### 8.1 CP+DeepEP: 缩小与 Round-Robin 差距

| 优化方向 | 预期收益 | 难度 | 说明 |
|---------|---------|------|------|
| DeepEP low-latency 模式 | 减少 notify 等待 | 低 | 需确保 prefill 不触发 buffer 限制 |
| 融合 dispatch 与 expert GEMM | 减少同步开销 | 高 | 将 dispatch 数据直接喂入 GEMM |
| 减少 AllGather 数据量 | 8.9%→6-7% | 高 | NSA attention 只 gather 需要的 KV blocks |
| 优化 DeepEP 3-barrier 协议 | 减少同步延迟 | 高 | 当前 3 barrier 是固定开销 |

### 8.2 CP Round-Robin: 进一步提升

| 优化方向 | 预期收益 | 难度 |
|---------|---------|------|
| fused_moe kernel 优化 | 21.7%→18% | 高 |
| gate GEMM + TopK 融合 | 减少 3.3% gatherTopK | 中 |
| moe_sum 融合 | 减少 2.1% | 中 |
| AllGather 算法 (TREE vs RING) | 减少尾部延迟 | 中 |

### 8.3 通用优化

| 优化方向 | 预期收益 |
|---------|---------|
| 更大模型验证 (60+ 层) | 优化效果更显著 |
| 不同并发度测试 | 找到各配置最优并发点 |
| Kernel 级 MoE 优化 | 突破计算瓶颈上限 |

---

## 9. nsys 报告文件索引

| 文件 | 场景 | 大小 | 日期 |
|------|------|------|------|
| `nocp_nocache.nsys-rep` | No-CP + 同 prompt | 608M | 2026-07-02 |
| `nocp_nocache_unique.nsys-rep` | No-CP + 不同 prompt | 363M | 2026-07-02 |
| `cp_rr_nocache.nsys-rep` | CP Round-Robin | 540M | 2026-07-03 |
| `cp_deepep_inseq_nocache.nsys-rep` | CP+DeepEP in-seq | 540M | 2026-07-03 |

旧报告 (2026-06-15~17, 不同配置, 仅供参考):
| `sglang_bench_sonnet.nsys-rep` | 无 DeepEP + cache | 1.4G | Jun 16 |
| `sglang_bench_deepep.nsys-rep` | DeepEP auto + cache | 1.2G | Jun 17 |

---

## 10. 相关文档

| 文档 | 路径 | 说明 |
|------|------|------|
| nsys + bench 使用手册 | `docs/sglang-nsys-bench-guide.md` | 如何跑 nsys profile 和 benchmark |
| nsys profile 详细分析 | `docs/nsys-profile-analysis.md` | CP RR vs DeepEP 的 kernel 级分析 |
| CP+MoE 优化记录 | `docs/sglang-cp-moe-optimization.md` | 优化尝试记录和代码改动 |
| 容器操作手册 | `docs/sglang-container-modify-guide.md` | 容器挂载和日常操作 |
| Benchmark Skill | `.claude/skills/sglang-bench.md` | 自动化 benchmark 流程 |

# SGLang + Nsight Systems Benchmark 使用文档

## 概述

本文档描述如何使用 `nsys launch` + `nsys start/stop` 方式对 sglang 模型服务进行 Nsight Systems 性能剖析，并在宿主机上运行 `vllm bench serve` 进行性能测试。

## 环境要求

| 组件 | 版本/路径 |
|------|-----------|
| Docker 容器 | `sglang-debug` (uhub.service.ucloud.cn/umirror/sglang:v0.5.12.post1-cu130) |
| sglang | 0.5.12.post1 |
| nsys (容器内) | 2026.2.1 |
| 宿主机 vllm 环境 | `/upfs/swan/miniconda3/envs/vllm/bin/python` (vllm 0.16.1rc1, 有 `vllm bench serve`) |
| 本地 Tokenizer | `/upfs/swan/models/GLM-5.1-FP8/GLM-5.1-FP8-10L` |
| Sonnet 文本 | `/upfs/swan/sonnet.txt` (~22,934 tokens) |
| GPU | 8x NVIDIA H20 (97GB each) |
| 容器网络模式 | host |
| 机器 IP | 117.50.175.64 |

## 快速开始

### 一、启动 sglang 服务（带 nsys launch）

```bash
# 1. 确认容器运行中
docker ps | grep sglang-debug

# 2. 确认 GPU 可用
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# 3. 启动 sglang，挂在 nsys launch 下
docker exec -d sglang-debug bash -c "nsys launch -t cuda,nvtx,osrt,cudnn,cublas \
  --session-new sglang_session \
  --trace-fork-before-exec true \
  -- python -m sglang.launch_server \
    --model-path /models \
    --trust-remote-code \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --tp-size 8 \
    --enable-nsa-prefill-context-parallel \
    --attn-cp-size 8 \
    --nsa-prefill-cp-mode round-robin-split"
```

### 二、等待服务就绪

```bash
# 实时轮询（不要用 sleep + 手动检查）
for i in $(seq 1 600); do
  if curl -sf http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
    echo "SERVER READY after ${i}s"
    break
  fi
  if ! docker exec sglang-debug ps aux | grep -q "[p]ython.*launch_server"; then
    echo "PROCESS DIED at ${i}s"
    docker exec sglang-debug find /tmp/nvidia/nsight_systems/ -name "pid_*_stderr.log" -mmin -5 -exec tail -30 {} \;
    break
  fi
  sleep 1
done
```

### 三、准备 Benchmark 数据集

**方式 1：合成 "The " 重复 prompt（~10000 tokens）**
```bash
python3 -c "
import json
for i in range(64):
    print(json.dumps({'prompt': 'The ' * 10000, 'output_tokens': 1}))
" > /tmp/bench_10000_1.jsonl
```

**方式 2：真实文本 sonnet.txt（~22934 tokens）**
```bash
python3 -c "
import json
with open('/upfs/swan/sonnet.txt') as f:
    text = f.read()
for i in range(64):
    print(json.dumps({'prompt': text, 'output_tokens': 1}))
" > /tmp/bench_sonnet_10k.jsonl
```

> **注意**：Custom dataset 格式必须是 `{"prompt": "...", "output_tokens": N}`，不是 `max_tokens`。

### 四、抓取 Profile 并跑 Benchmark

```bash
# 1. 开始 nsys 抓取（不加 --stats=true 可加速 stop）
docker exec sglang-debug nsys start --session sglang_session -c none \
  -o /tmp/sglang_bench -f true

# 2. 从宿主机跑 benchmark（3 次）
for i in $(seq 1 3); do
  echo "=== Run $i ==="
  /upfs/swan/miniconda3/envs/vllm/bin/vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 \
    --port 30000 \
    --max-concurrency 64 \
    --num-prompts 64 \
    --dataset-name custom \
    --dataset-path /tmp/bench_sonnet_10k.jsonl \
    --model "/models" \
    --custom-output-len 1 \
    --tokenizer /upfs/swan/models/GLM-5.1-FP8/GLM-5.1-FP8-10L \
    --trust-remote-code
done

# 3. 停止 nsys 抓取（会自动生成 .nsys-rep 报告，可能需要 5-15 分钟）
docker exec sglang-debug nsys stop --session sglang_session

# 4. 等待报告生成完成后拷贝到宿主机
docker exec sglang-debug ls -lh /tmp/sglang_bench.nsys-rep
docker cp sglang-debug:/tmp/sglang_bench.nsys-rep /upfs/swan/sglang_bench.nsys-rep
```

### 五、传输报告到其他机器

```bash
# 在目标机器上执行（显示进度 + 断点续传）
rsync -avP root@117.50.175.64:/upfs/swan/sglang_bench.nsys-rep ./

# 如果 SSH 端口非 22：
rsync -avP -e 'ssh -p 端口' root@117.50.175.64:/upfs/swan/sglang_bench.nsys-rep ./
```

**已生成的报告文件：**

| 报告文件 | 大小 | 配置 |
|----------|------|------|
| `sglang_bench_fork.nsys-rep` | 90MB | 无 DeepEP，1 run |
| `sglang_bench_fork2.nsys-rep` | 95MB | 无 DeepEP，1 run |
| `sglang_bench_fork_3runs.nsys-rep` | 137MB | 无 DeepEP，3 runs |
| `sglang_bench_vllm_3runs.nsys-rep` | 1.2GB | 无 DeepEP，vllm bench 3 runs |
| `sglang_bench_sonnet.nsys-rep` | 1.4GB | 无 DeepEP，sonnet.txt 3 runs |
| `sglang_bench_deepep.nsys-rep` | 1.2GB | DeepEP auto，sonnet.txt 3 runs |

传输示例（以 DeepEP 报告为例）：
```bash
rsync -avP root@117.50.175.64:/upfs/swan/sglang_bench_deepep.nsys-rep ./
```

### 六、重复 Benchmark（服务不重启）

```bash
docker exec sglang-debug nsys start --session sglang_session -c none \
  -o /tmp/sglang_bench_run2 -f true

/upfs/swan/miniconda3/envs/vllm/bin/vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 64 \
  --num-prompts 64 \
  --dataset-name custom \
  --dataset-path /tmp/bench_sonnet_10k.jsonl \
  --model "/models" \
  --custom-output-len 1 \
  --tokenizer /upfs/swan/models/GLM-5.1-FP8/GLM-5.1-FP8-10L \
  --trust-remote-code

docker exec sglang-debug nsys stop --session sglang_session
docker cp sglang-debug:/tmp/sglang_bench_run2.nsys-rep /upfs/swan/sglang_bench_run2.nsys-rep
```

## 核心架构

```
┌─────────────────────────────────────────────────┐
│                  宿主机 (Host)                    │
│                                                   │
│  ┌──────────────┐     ┌────────────────────────┐ │
│  │ vllm conda   │────>│  http://127.0.0.1:30000 │ │
│  │ bench serve  │     │  (host network 直接访问)  │ │
│  └──────────────┘     └──────────┬─────────────┘ │
│                                  │                 │
│  ┌───────────────────────────────┼──────────────┐ │
│  │        容器 sglang-debug       │              │ │
│  │                               │              │ │
│  │  ┌─────────┐   ┌─────────────▼───────────┐  │ │
│  │  │  nsys    │   │  sglang launch_server    │  │ │
│  │  │  launch  │──>│  (8 GPUs, tp=8, cp=8)   │  │ │
│  │  └─────────┘   └─────────────────────────┘  │ │
│  │                                              │ │
│  │  nsys start/stop 控制 profile 抓取时间段      │ │
│  └──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## nsys 工作流程

| 阶段 | 命令 | 说明 |
|------|------|------|
| 启动监控 | `nsys launch` | 启动 sglang，挂载 nsys 跟踪，**不抓取数据** |
| 开始抓取 | `nsys start --session sglang_session` | 开始记录 CUDA/NVTX 等事件 |
| 跑 benchmark | 宿主机上跑 `vllm bench serve` | 数据发送到 sglang 服务 |
| 结束抓取 | `nsys stop --session sglang_session` | 停止记录，自动生成 .nsys-rep 报告 |

## 关键参数说明

### nsys launch 参数

| 参数 | 说明 |
|------|------|
| `-t cuda,nvtx,osrt,cudnn,cublas` | 跟踪类型：CUDA API、NVTX、OS Runtime、cuDNN、cuBLAS |
| `--session-new sglang_session` | 创建命名 session，后续 start/stop 引用 |
| `--trace-fork-before-exec true` | **必须**：跟踪 fork 后 exec 前的子进程（sglang 会 fork worker） |

### nsys start 参数

| 参数 | 说明 |
|------|------|
| `--session sglang_session` | 指向 launch 创建的 session |
| `-c none` | 不用 cudaProfilerApi/nvtx 触发，立即开始抓取 |
| `-o /tmp/xxx` | 输出报告文件名 |
| `-f true` | 覆盖已有报告 |
| `--stats=true` | 生成统计摘要（可选，会显著增加 stop 时间） |

### vllm bench serve 关键参数

| 参数 | 说明 |
|------|------|
| `--backend vllm` | 使用 vllm/OpenAI completions 接口 |
| `--tokenizer <path>` | **必须**：本地 tokenizer 路径，否则 token 统计不准 |
| `--trust-remote-code` | **必须**：加载自定义 tokenizer |
| `--custom-output-len 1` | 覆盖 dataset 中的 output_tokens |
| `--dataset-name custom` | 使用自定义 JSONL 数据集 |
| `--dataset-path <file>` | JSONL 数据集路径 |
| `--max-concurrency 64` | 最大并发请求数 |
| `--num-prompts 64` | 总请求数 |
| `--model "/models"` | 模型名（必须与服务端一致） |

### sglang 服务参数

| 参数 | 说明 |
|------|------|
| `--tp-size 8` | 8 卡张量并行 |
| `--enable-nsa-prefill-context-parallel` | 启用 NSA prefill context parallel |
| `--attn-cp-size 8` | 8 卡 attention context parallel |
| `--nsa-prefill-cp-mode round-robin-split` | CP 模式：round-robin 分割 |

### DeepEP a2a 参数（减少 NCCL 通信瓶颈）

在 CP+NSA 配置下，NCCL AllGather/ReduceScatter 占 GPU 时间 ~96%，实际计算仅 ~3%。DeepEP a2a 模式让 MoE 走 all-to-all dispatch，替代 AllGather/ReduceScatter 路径，可显著降低通信开销。

| 参数 | 说明 |
|------|------|
| `--moe-a2a-backend deepep` | 启用 DeepEP MoE all-to-all 后端 |
| `--ep-size 8` | Expert parallel 大小（需与 tp-size 一致，sglang 自动调整） |
| `--deepep-mode auto` | **推荐**：prefill 走 normal（大 buffer），decode 走 low_latency（低延迟） |
| `--cuda-graph-max-bs 128` | **必须**：限制 decode cuda graph capture 批次 ≤ 128（low_latency buffer 上限） |

**⚠️ DeepEP 踩坑：**

1. **不要用 `--deepep-mode low_latency`**：low_latency 的 dispatch buffer 硬编码为 `num_max_dispatch_tokens_per_rank=128`，prefill 大 batch（如 22934 tokens）远超此限制会触发 DeepEP 断言失败：
   ```
   RuntimeError: Assertion error deep_ep.cpp:1105
   'x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank'
   ```
   必须用 `auto`（默认值），让 prefill 走 normal 模式（大 buffer）。

2. **必须加 `--cuda-graph-max-bs 128`**：即使 `--deepep-mode auto`，decode 仍走 low_latency（128 token/rank 上限）。默认 `cuda_graph_max_bs=512` 会在 capture 阶段超过 128 导致同样的断言失败。

3. **CP 要求 DeepEP**：sglang 源码中 `DeepseekV4DecoderLayer` 有断言 `CP requires DeepEP (moe_a2a_backend == deepep)`，CP 的 per-rank token split 只有 DeepEP 测试过。

**带 DeepEP 的完整启动命令：**
```bash
docker exec -d sglang-debug bash -c "nsys launch -t cuda,nvtx,osrt,cudnn,cublas \
  --session-new sglang_session \
  --trace-fork-before-exec true \
  -- python -m sglang.launch_server \
    --model-path /models \
    --trust-remote-code \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --tp-size 8 \
    --enable-nsa-prefill-context-parallel \
    --attn-cp-size 8 \
    --nsa-prefill-cp-mode round-robin-split \
    --moe-a2a-backend deepep \
    --ep-size 8 \
    --deepep-mode auto \
    --cuda-graph-max-bs 128"
```

## 查看报告

1. 将 `.nsys-rep` 文件传输到本地
2. 使用 [Nsight Systems GUI](https://developer.nvidia.com/nsight-systems) 打开
3. 重点关注：
   - CUDA API 时间线（各 GPU 的 kernel 执行）
   - NCCL 通信（AllGather / ReduceScatter）
   - NVTX 标记（如果 sglang 有标注）
   - Memory 操作（H2D / D2H / D2D）

## 性能分析参考

在当前配置（10 层 MoE + tp=8 + attn-cp-size=8）下，NCCL 通信是主要瓶颈：

| GPU 时间分布 | 占比 |
|-------------|------|
| NCCL AllGather | ~73% |
| NCCL ReduceScatter | ~22% |
| 实际计算（attention + MoE） | ~3% |

原因：10 层小模型计算量不足以掩盖 8 卡 CP 的通信开销。

**Benchmark 参考结果 - 无 DeepEP**（Sonnet ~22934 tokens in / 1 token out，64 并发）：

| Run | Duration | Throughput | Mean TTFT | Total tok/s |
|-----|----------|-----------|-----------|------------|
| 1（冷启动） | 14.72s | 4.35 req/s | 14,575ms | 99,721 |
| 2 | 9.22s | 6.94 req/s | 2,209ms | 159,225 |
| 3 | 3.75s | 17.06 req/s | 2,542ms | 391,250 |

**Benchmark 参考结果 - DeepEP auto**（相同配置 + `--moe-a2a-backend deepep --ep-size 8 --deepep-mode auto --cuda-graph-max-bs 128`）：

| Run | Duration | Throughput | Mean TTFT | Total tok/s | 失败请求 |
|-----|----------|-----------|-----------|------------|---------|
| 1 | 3.14s | 20.41 req/s | 2,082ms | 468,147 | 0 |
| 2 | 3.17s | 20.20 req/s | 2,152ms | 463,465 | 0 |
| 3 | 3.04s | 21.03 req/s | 2,093ms | 482,392 | 0 |

**对比（Run 3 最佳成绩）：**

| 配置 | Total tok/s | Mean TTFT | 提升 |
|------|------------|-----------|------|
| 无 DeepEP | 391,250 | 2,542ms | 基线 |
| DeepEP auto | 482,392 | 2,093ms | **+23% 吞吐 / -18% TTFT** |

DeepEP a2a 将 MoE 通信从 NCCL AllGather/ReduceScatter 替换为 all-to-all dispatch，吞吐从 ~391k 提升到 ~482k tok/s，TTFT 从 2542ms 降到 2093ms。

### Nsight Systems Profile 对比分析

通过 nsys stats 对比 `sglang_bench_sonnet.nsys-rep`（无 DeepEP）和 `sglang_bench_deepep.nsys-rep`（DeepEP auto）的 CUDA kernel 时间分布：

**GPU Kernel 时间对比：**

| Kernel 类别 | 无 DeepEP | DeepEP auto | 变化 |
|------------|----------|-------------|------|
| NCCL AllGather | 48.6% (20,518ms) | 50.5% (17,379ms) | **-3,139ms (-15.3%)** |
| NCCL ReduceScatter | 28.6% (12,067ms) | — (完全消除) | **-12,067ms (-100%)** |
| DeepEP dispatch | — | 10.9% (3,746ms) | 新增 |
| DeepEP combine | — | 3.7% (1,267ms) | 新增 |
| DeepEP 小 kernel (layout/gather/scatter) | — | ~1.0% (~344ms) | 新增 |
| DeepGEMM MoE GEMM | 4.0% (1,693ms) | 5.8% (2,001ms) | +308ms |
| Sparse Attention | 4.5% (1,907ms) | 5.4% (1,855ms) | -52ms |
| DeepGEMM MQA logits | 4.7% (2,005ms) | 6.1% (2,087ms) | +82ms |

**优化效果总结：**

| 项目 | 节省/增加 | 说明 |
|------|----------|------|
| ReduceScatter 消除 | **-12,067ms** | 最大优化点，MoE 不再走 ReduceScatter |
| AllGather 减少 | **-3,139ms** | MoE 结果不再需要 AllGather，仅 attention CP 保留 |
| DeepEP 新增开销 | +5,013ms | all-to-all dispatch + combine，但比 ReduceScatter 轻量 |
| **净省通信时间** | **~10,193ms** | 通信时间减少约 32% |

**结论：** DeepEP 的核心优化是**用更轻量的 all-to-all dispatch+combine（5 秒）替代了昂贵的 NCCL ReduceScatter（12 秒）**，同时 AllGather 也因 MoE 路径改变而减少。省出的 ~10 秒通信时间就是吞吐从 391k 提升到 482k tok/s 的来源。AllGather 仍然存在（用于 attention CP 通信），仍是当前瓶颈。

## 常见问题

### Q: nsys start 报 "Connection refused"
A: nsys daemon 状态异常，通常是因为之前有卡住的 session。重启容器：
```bash
docker restart sglang-debug
# 然后重新 nsys launch 启动 sglang
```

### Q: sglang 启动后看不到日志
A: `nsys launch` 会捕获 stdout/stderr，日志在：
```bash
docker exec sglang-debug find /tmp/nvidia/nsight_systems/ -name "pid_*_stderr.log" | xargs tail -20
```

### Q: vllm bench serve 显示 Total input tokens: 1
A: 没有指定 `--tokenizer` 参数。必须加 `--tokenizer /upfs/swan/models/GLM-5.1-FP8/GLM-5.1-FP8-10L --trust-remote-code`，否则 vllm 无法 tokenize 统计 token 数。

### Q: custom dataset 格式报错
A: 格式必须是 `{"prompt": "...", "output_tokens": 1}`，不是 `max_tokens`。这是 vllm 的 CustomDataset 类要求的。

### Q: nsys stop 很慢
A: 3次 × 64并发 × 10k+ token 的数据量很大（1~1.5GB nsys-rep），报告生成是单线程 CPU 密集操作。优化方法：
- 去掉 `--stats=true`，省掉 sqlite 生成
- 每次只抓 1 次 benchmark
- 用 `--defer-report` 延迟生成

### Q: 报告文件没生成
A: `nsys stop` 后需要等待报告生成完成（5-15分钟）。检查：
```bash
docker exec sglang-debug nsys sessions list  # Generation = 还在生成，Launched = 完成
docker exec sglang-debug ls -lh /tmp/你的报告名.nsys-rep
```

### Q: 只用了1张卡
A: 检查是否误设了 `CUDA_VISIBLE_DEVICES`。多卡场景（`--tp-size` 或 `--attn-cp-size > 1`）不要限制可见 GPU。

### Q: 如何只抓 benchmark 不抓启动
A: 服务启动完成后再 `nsys start`，跑完 benchmark 后 `nsys stop`。这样 profile 里只有推理过程。

### Q: zombie 进程太多
A: 多次 nsys launch/stop 会积累 zombie。重启容器清理：
```bash
docker restart sglang-debug
```

### Q: DeepEP 启动报错 `deep_ep.cpp:1105 x.size(0) <= num_max_dispatch_tokens_per_rank`
A: DeepEP low_latency 模式的 dispatch buffer 硬编码为 128 tokens/rank。两个原因会触发：
1. 用了 `--deepep-mode low_latency`（强制所有操作走 low_latency），prefill 大 batch 超限 → **改用 `--deepep-mode auto`**
2. 默认 `cuda_graph_max_bs=512`，decode cuda graph capture 时 batch > 128 → **加 `--cuda-graph-max-bs 128`**

正确配置：`--deepep-mode auto --cuda-graph-max-bs 128`

### Q: 启动时 GPU0 100% 其他卡 0%
A: DeepGEMM warmup 阶段（JIT 编译 FP8 GEMM kernel）在 rank 0 (GPU0) 上串行执行，其他卡在等。日志会显示 `DeepGEMM warmup: XX%|...`。这是正常启动现象，warmup 完成后 8 卡会一起工作。warmup 通常需要 1-2 分钟。

### Q: 报告太大传输慢
A: 使用 rsync 断点续传：
```bash
rsync -avP root@117.50.175.64:/upfs/swan/sglang_bench.nsys-rep ./
```

## 注意事项

1. **不要用 `nsys profile`**：它会在 `--duration` 超时后杀掉 sglang 进程
2. **不要在容器内跑 benchmark**：容器里没装 vllm，从宿主机跑
3. **不要轻易关服务**：重复 benchmark 不需要重启服务
4. **`--trace-fork-before-exec true` 只能在 `nsys launch` 时设置**：不能在 `nsys start` 时加
5. **`--cpuctxsw` 不能用在 `nsys launch`**：已废弃，需要在 `nsys start` 时设置（但一般不需要）
6. **必须指定 `--tokenizer`**：否则 vllm bench serve 的 token 统计不准
7. **custom dataset 用 `output_tokens`**：不是 `max_tokens`
8. **使用 `/upfs/swan/miniconda3/envs/vllm/bin/python`**：这个环境有 `vllm bench serve` 命令，vllm2 环境没有
9. **DeepEP 不要用 `--deepep-mode low_latency`**：low_latency buffer 只有 128 tokens/rank，prefill 大 batch 会断言失败。用 `auto`（默认值，prefill 走 normal，decode 走 low_latency）
10. **DeepEP 必须加 `--cuda-graph-max-bs 128`**：decode cuda graph capture 不能超过 low_latency 的 128 token/rank 上限
11. **启动时 GPU0 满载其他卡 0% 是正常的**：DeepGEMM warmup 在 rank 0 串行执行，等进度条跑完即可

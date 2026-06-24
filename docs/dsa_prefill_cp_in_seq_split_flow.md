> SGLang DSA Prefill Context Parallel (In-Seq-Split) 完整流程
> **配置参数**  
> `--enable-dsa-prefill-context-parallel --attn-cp-size 8 --dsa-prefill-cp-mode in-seq-split`  
>  
> 注：`--enable-nsa-prefill-context-parallel` / `--nsa-prefill-cp-mode` 为已废弃的别名  
> （`DeprecatedStoreTrueAction`，[server_args.py:6820](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L6820) / `DeprecatedAliasStoreAction`，[server_args.py:6835](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L6835)），  
> 当前推荐使用上述新名称。
>
> **适用模型**：本文档主要以 DeepSeek V4（`DeepseekV4ForCausalLM`，`deepseek_v4.py`）为例描述。
> 同一套 CP 参数也适用于走 `DeepseekV2ForCausalLM`（`deepseek_v2.py`）代码路径的 DSA 模型，包括
> **GLM DSA**（`GlmMoeDsaForCausalLM`，[glm4_moe.py:1481](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/glm4_moe.py#L1481)，继承自 `DeepseekV2ForCausalLM`）、
> DeepSeek V3.2（`DeepseekV32ForCausalLM`）等。V2 路径与 V4 路径的差异在各 Step 中以
> "**▶ GLM DSA / V2 路径差异**"标注。
>
> **对比文档**：本文对应 `in-seq-split`（zigzag 两端分割）模式；`round-robin-split`（stride 交错分割）模式见
> [dsa_prefill_cp_round_robin_split_flow.md](dsa_prefill_cp_round_robin_split_flow.md)。
> 两模式共享同一套入口与大部分基础设施，差异点在各 Step 中以
> "**▶ vs round-robin-split**" 标注。

---

#### Step 0: 参数初始化与校验

服务启动时，`ServerArgs.__post_init__()`（[server_args.py:887](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L887)）经过一系列处理后，在 [server_args.py:1000](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1000) 调用 `_handle_context_parallelism()` 方法（定义于 [server_args.py:3218](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3218)），校验：

- `attn_cp_size = 8`，要求 `tp_size % attn_cp_size == 0`（[server_args.py:3234](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3234)）
- `enable_dsa_prefill_context_parallel = True`（字段声明 [server_args.py:803](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L803)）
- `dsa_prefill_cp_mode = "in-seq-split"`（字段声明 [server_args.py:804](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L804)，合法值 `["in-seq-split", "round-robin-split"]`，定义于 `DSA_PREFILL_CP_SPLIT_CHOICES`，[server_args.py:275](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L275)）
- 两个 flag `enable_prefill_context_parallel` 和 `enable_dsa_prefill_context_parallel` 互斥（[server_args.py:3219-3230](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3219-L3230)）

字段声明（[server_args.py:803-804](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L803-L804)）：

```python
enable_dsa_prefill_context_parallel: bool = False            # 是否启用 DSA prefill CP，默认关闭
dsa_prefill_cp_mode: str = "round-robin-split"               # CP 分割模式，默认 round-robin-split
                                                              # in-seq-split 需用户显式指定
```

`_handle_context_parallelism()`（[server_args.py:3218](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3218)）核心校验逻辑：

```python
def _handle_context_parallelism(self):                       # CP 参数校验入口
    if (
        self.enable_prefill_context_parallel                 # 普通 MLA CP（V3/R1）
        and self.enable_dsa_prefill_context_parallel          # DSA CP（V3.2/GLM）
    ):                                                       # 两种 CP 不能同时启用
        raise ValueError(
            "--enable-prefill-context-parallel and "
            "--enable-nsa-prefill-context-parallel are mutually exclusive. ..."
        )

    if self.attn_cp_size > 1:                                # 当 CP size > 1 时校验整除关系
        assert (
            self.tp_size % self.attn_cp_size == 0            # TP size 必须能被 CP size 整除
        ), "tp_size must be divisible by attn_cp_size"
        assert (
            self.tp_size % (self.dp_size * self.attn_cp_size) == 0  # 也要能被 DP×CP 整除
        ), "tp_size must be divisible by dp_size * attn_cp_size"
```

每个 `TpModelWorker` 初始化时会拿到自己的 `attn_cp_rank`（0~7），并通过 `get_attention_cp_rank()`（[dp_attention.py:338](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L338)）/ `get_attention_cp_size()`（[dp_attention.py:342](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L342)）获取当前 rank 和总 CP size：

```python
def get_attention_cp_rank() -> int:                          # 获取当前 GPU 在 CP group 中的 rank（0~7）
    return get_attn_context_model_parallel_rank()             # 底层从 NCCL process group 获取

def get_attention_cp_size() -> int:                          # 获取 CP group 的总大小（如 8）
    return get_attn_context_model_parallel_world_size()       # 底层从 NCCL process group 获取
```

##### 0a. in-seq-split 的自动强制配置（关键差异）

`_handle_model_specific_adjustments()`（定义于 [server_args.py:1792](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1792)，DSA-CP 子块 `:1857-1893`）中，当 `dsa_prefill_cp_mode == "in-seq-split"` 时，**自动强制**设置 MoE 后端为 DeepEP EP（[server_args.py:1861-1869](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1861-L1869)）：

```python
if self.enable_dsa_prefill_context_parallel:
    logger.warning("Context parallel feature is still under experiment. ...")
    if self.dsa_prefill_cp_mode == "in-seq-split":           # in-seq-split 分支
        # TODO Supports moe_dense_tp_size != 1, kv cache dtype = "fp8",
        #      moe_a2a_backend non-deepep and cross-machine operation.
        self.enable_dp_attention = True                       # 强制开 DP attention
        self.moe_dense_tp_size = 1                            # dense TP=1
        self.moe_a2a_backend = "deepep"                       # ★ 强制 MoE 后端 = DeepEP
        self.ep_size = self.tp_size                           # ★ 强制 EP size = TP size
        logger.warning(
            "For in-seq split mode, we have the following restrictions: "
            "moe_dense_tp_size == 1, moe_a2a_backend == deepep, "
            "ep_size == tp_size, batch_size == 1"
        )
    else:                                                    # round-robin-split 分支
        self.enable_dp_attention = True
        self.moe_dense_tp_size = 1
        assert self.dp_size == 1, "For round-robin split mode, dp attention is not supported."
        # 注：不自动设 moe_a2a_backend 和 ep_size
    assert self.tp_size <= 8, "Context parallel only supports single machine (tp_size <= 8). ..."
    self.attn_cp_size = self.tp_size // self.dp_size
    self.disable_piecewise_cuda_graph = True
```

**▶ vs round-robin-split**：round-robin-split **不**自动设 `moe_a2a_backend` / `ep_size`，保留默认 `moe_a2a_backend="none"`、`ep_size=1`，允许 TP MoE、multi-batch、FP8 KV cache。in-seq-split 强制 DeepEP EP + `batch_size == 1`，且不支持 FP8 KV cache（TODO）。

由此推导出的并行参数（[distributed/parallel_state.py:1931](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L1931) / `:2003`）：

```
attn_tp_size = tp_size // attn_cp_size // attn_dp_size = 8 // 8 // 1 = 1   # attention 权重不切分
moe_tp_size  = tp_size // moe_ep_size  // moe_dp_size  = 8 // 8  // 1 = 1   # MoE expert 权重不切分（EP 模式）
```

即：每个 GPU 持有全部 attention heads（权重复制），但只持有 `1/cp_size` 个完整 expert（EP 模式，权重不切分）。

**▶ GLM DSA / V2 路径差异**：`GlmMoeDsaForCausalLM` 的 `architecture` 为 `"GlmMoeDsaForCausalLM"`，在 `_handle_model_specific_adjustments()`（[server_args.py:1828-1835](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1828-L1835)）中与 `DeepseekV32ForCausalLM` 等一起处理：

```python
if model_arch in [                                           # 检查模型架构是否属于 MLA 族
    "DeepseekV3ForCausalLM",                                 # DeepSeek V3
    "DeepseekV32ForCausalLM",                                # DeepSeek V3.2
    "KimiK25ForConditionalGeneration",                       # Kimi K2.5
    "MistralLarge3ForCausalLM",                              # Mistral Large 3
    "PixtralForConditionalGeneration",                       # Pixtral（多模态）
    "GlmMoeDsaForCausalLM",                                 # GLM 5.1 DSA
]:
    if is_deepseek_dsa(hf_config):                           # 进一步判断是否为 DSA 模型（需 index_topk）
        # ...设置 attention_backend = "dsa"，配置 dense attn threshold 等
```

`is_deepseek_dsa()`（[model_config.py:102](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/configs/model_config.py#L102)）对其返回 `True`（需同时满足 config 中存在 `index_topk` 属性）。in-seq-split 的自动强制配置逻辑完全相同，无差异。

---

#### Step 1: HTTP 请求到达 TokenizerManager

用户 → `POST /v1/chat/completions {text: "hello"}`

1. `TokenizerWorker`（HTTP worker 进程）收到请求
2. Tokenizer 对 `"hello"` 做 tokenize，得到 `input_ids`（比如 `[15496]`，取决于具体 tokenizer）
3. 构造 `TokenizedGenerateReqInput`（[managers/io_struct.py:733](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/io_struct.py#L733)，包含 `input_ids`、`sampling_params` 等）
4. 通过 ZMQ（`send_pyobj`）发送给 `MultiTokenizerRouter` → 转发给 Scheduler

   请求发送路径：`TokenizerManager` 通过 `send_to_scheduler` ZMQ socket 发出请求。  
   结果接收路径：`handle_loop()`（[tokenizer_manager.py:1759](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/tokenizer_manager.py#L1759)）是 `TokenizerManager` 的主事件循环：

   ```python
   async def handle_loop(self):                                # TokenizerManager 的主事件循环
       """The event loop that handles requests"""
       while True:                                             # 持续轮询
           with self.soft_watchdog.disable():                  # 暂停 watchdog 计时（等待期间不触发超时）
               # ... 从 detokenizer 接收 BatchStrOutput / BatchEmbeddingOutput / BatchTokenIDOutput
   ```

---

#### Step 2: Scheduler 接收请求

`Scheduler.recv_requests()`（[scheduler_components/request_receiver.py:65](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/scheduler_components/request_receiver.py#L65)）从 ZMQ 接收 `TokenizedGenerateReqInput`：

```python
def recv_requests(self) -> List[...]:                        # Scheduler 接收请求的入口
    """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
    if self.recv_skipper is not None:                         # 如果有请求跳过器（限流/跳过机制）
        if not self.recv_skipper.handle(self.get_last_forward_mode()):
            return []                                         # 跳过本轮接收
    recv_reqs = self._pull_raw_reqs()                        # 从 ZMQ socket 拉取原始请求
    recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs) # 将请求广播给所有 TP rank
    recv_reqs = self._apply_mm_receiver(recv_reqs)           # 处理多模态（图片/视频）输入
    self._finalize_shm_features(recv_reqs)                   # 完成共享内存特征的传输
    return recv_reqs                                          # 返回本轮接收的所有请求
```

请求通过类型分发表路由到对应处理函数（[scheduler.py:1281-1283](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/scheduler.py#L1281-L1283)）：

```python
self._request_dispatcher = TypeBasedDispatcher([             # 按请求类型分发到对应处理函数
    (TokenizedGenerateReqInput, self.handle_generate_request),  # 生成请求 → handle_generate
    (TokenizedEmbeddingReqInput, self.handle_embedding_request),# 嵌入请求 → handle_embedding
    # ...                                                      # 还有 batch 类型等
])
```

`handle_generate_request()`（[scheduler.py:1899](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/scheduler.py#L1899)）核心流程：

```python
def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
    # ...                                                     # 前置处理：session 路由等
    req = Req(                                               # 构造内部请求对象 Req
        recv_req.rid,                                        # 请求唯一 ID
        recv_req.input_text,                                 # 原始文本
        recv_req.input_ids,                                  # tokenize 后的 token ID 列表
        recv_req.sampling_params,                            # 采样参数（temperature, top_p 等）
        return_logprob=recv_req.return_logprob,              # 是否返回 logprob
        # ... 更多参数
    )
    # ...校验、grammar 处理后：
    self._add_request_to_queue(req)                          # 将请求加入等待队列
```

`_add_request_to_queue()`（[scheduler.py:2157](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/scheduler.py#L2157)）将请求加入等待队列：

```python
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    if not self._set_or_validate_priority(req):              # 校验/设置请求优先级
        return                                                # 优先级无效则丢弃
    # ...                                                     # 排队限制检查、KV cache 预取
    self.waiting_queue.append(req)                           # 加入等待队列尾部
    req.time_stats.set_wait_queue_entry_time()               # 记录入队时间（用于延迟统计）
```

---

#### Step 3: Scheduler 组 Batch（关键 —— in-seq-split 强制单 batch）

Scheduler 主循环调用 `get_new_batch_prefill()` → `_get_new_batch_prefill_raw()`（[scheduler.py:2553](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/scheduler.py#L2553)）：

```python
def _get_new_batch_prefill_raw(self, ...) -> Optional[ScheduleBatch]:
    # 检查 grammar queue（结构化输出的 grammar 编译完成后回到主队列）
    if self.grammar_manager.has_waiting_grammars():           # 有等待中的 grammar 请求
        ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
        for req in ready_grammar_requests:                    # 将编译完成的请求
            self._add_request_to_queue(req)                   # 重新放回等待队列
    # ...
    if (
        self.running_batch.batch_is_full or len(self.waiting_queue) == 0  # batch 满或无等待请求
    ) and self.chunked_req is None:                           # 且没有分块未完成的请求
        return None                                           # 无法组成新 batch
    # ...优先级排序、PrefillAdder 逐个添加请求
```

##### in-seq-split 的单 batch 强制（`PrefillAdder.add_one_req`）

`PrefillAdder.__init__()`（[schedule_policy.py:489](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/schedule_policy.py#L489)）缓存 in-seq-split 标志：

```python
self.dsa_prefill_cp_in_seq_split = is_dsa_prefill_cp_in_seq_split()  # 是否为 in-seq-split 模式
```

`add_one_req()`（[schedule_policy.py:845-861](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/schedule_policy.py#L845-L861)）中，in-seq-split 模式强制 prefill batch 只能含 1 个请求：

```python
def add_one_req(self, req, has_chunked_req, truncation_align_size):
    # ...
    # TODO support cp with multiple requests
    # Enabling context parallelism currently presents precision issues;
    # therefore, the prefill-batch setting is temporarily set to 1.
    if (self.dsa_prefill_cp_in_seq_split) and len(self.can_run_list) >= 1:  # in-seq-split 且已有 1 个请求
        return AddReqResult.OTHER                            # 拒绝再加入第 2 个请求
    # ...
```

**▶ vs round-robin-split**：round-robin-split **无此限制**，支持 multi-batch prefill（`dsa_cp_round_robin_split_q_seqs` 处理多 seq 的 stride 分发与 `bs_idx` 映射）。in-seq-split 因 zigzag prev/next 分段 topk 的多 batch 支持有精度问题（[dsa_indexer.py:1497](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1497) 的 `# TODO support mutil-batch`），暂强制 `batch_size == 1`。

1. `PrefillAdder` 从 `waiting_queue` 中取出请求
2. 调用 `req.init_next_round_input(tree_cache)` 做 radix cache 前缀匹配
3. 调用 `adder.add_one_req()` 判断能否加入 batch（in-seq-split 下第 2 个起被拒）
4. 得到 `can_run_list`（本 batch 要 prefill 的请求列表，in-seq-split 下长度为 1）
5. 创建 `ScheduleBatch`，调用 `new_batch.prepare_for_extend()`

`prepare_for_extend()`（[schedule_batch.py:1823](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/schedule_batch.py#L1823)）设置 forward mode 并准备 tensor：

```python
def prepare_for_extend(self):                                # 准备 prefill/extend 所需的 tensor
    self.forward_mode = ForwardMode.EXTEND                   # 设置模式为 EXTEND（prefill）
    # Init tensors
    reqs = self.reqs                                         # 本 batch 中的所有请求
    input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]  # 去掉 cache 命中的前缀，取需要计算的 token
    extend_num_tokens = sum(len(ids) for ids in input_ids)   # 本 batch 总共需要 prefill 的 token 数
    seq_lens = [len(r.fill_ids) for r in reqs]               # 每个请求的完整序列长度（含 prefix）
    prefix_lens = [len(r.prefix_indices) for r in reqs]      # 每个请求命中 radix cache 的前缀长度
    extend_lens = [r.extend_input_len for r in reqs]         # 每个请求实际需要 extend 的长度
    # ... 构建 pinned tensor，准备 H2D 传输
```

   此时 `ScheduleBatch.forward_mode = ForwardMode.EXTEND`

---

#### Step 4: CP 元数据准备（CausalLM.forward 入口 —— 生成 zigzag 索引）

`TpModelWorker.forward_batch_generation()`（[tp_worker.py:447](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/tp_worker.py#L447)）将 `ScheduleBatch` 转换为 `ForwardBatch` 并调用模型 forward：

```python
def forward_batch_generation(                                # TP worker 的生成入口
    self, batch: Optional[ScheduleBatch],
    forward_batch: Optional[ForwardBatch] = None, ...
) -> GenerationBatchResult:
    if batch is not None:                                    # 有 ScheduleBatch 时
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)  # 转换为 GPU 侧的 ForwardBatch
    # ...
    if self.pp_group.is_last_rank:                           # 仅 pipeline 最后一个 stage 执行 forward
        out = self.model_runner.forward(                     # 调用模型 forward（最终进入 CausalLM.forward）
            forward_batch, pp_proxy_tensors=pp_proxy_tensors,
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph  # 取出 logits 和 CUDA graph 标记
```

最终进入模型的 `forward()` 方法。

**DeepSeek V4 路径**：进入 `DeepseekV4ForCausalLM.forward()`（[deepseek_v4.py:1722](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1722)）。

在此方法中，首先检查是否启用 CP 并准备元数据（[deepseek_v4.py:1730-1748](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1730-L1748)）：

```python
if self.dsa_enable_prefill_cp:                               # 如果启用了 DSA CP
    if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):  # 检查序列长度是否满足 CP 要求
        forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(  # 构建 CP 元数据
            len(input_ids), self.cp_rank, self.cp_size,      # 序列长度、当前 rank、CP 总数
            forward_batch.seq_lens_cpu.tolist(),              # 每个请求的序列长度
            extend_seqs_len=forward_batch.extend_seq_lens_cpu,  # 每个请求需要 extend 的长度
        )
        if is_dsa_prefill_cp_round_robin_split():            # round-robin-split 才走 reindex 分支
            attn_backend = get_attn_backend()                # 获取当前 attention backend（如 DSA backend）
            metadata = attn_backend.forward_metadata         # 取出前向元数据
            core_meta = metadata.core_attn_metadata          # 取出核心 attention 元数据
            core_meta.apply_cp_reindex()                     # 对 attention index 做 CP 重排（仅 RR）
            core_meta.init_flashmla_related(is_prefill=True) # 初始化 FlashMLA 相关参数（仅 RR）
            # ...                                             # 还会重建 indexer_metadata
```

**▶ vs round-robin-split**：in-seq-split **不**执行 `apply_cp_reindex()` / `init_flashmla_related()`（这两者只对 round-robin-split 生效）。in-seq-split 的 CP 切分信息全部编码在 `attn_cp_metadata`（`ContextParallelMetadata`）的 zigzag 索引字段中，由后续 `cp_split_and_rebuild_data` / `cp_all_gather_rerange_output` / indexer 消费。

##### 4a. `can_dsa_cp_split()` 门槛（[dsa/utils.py:175](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L175)）

```python
def can_dsa_cp_split(seq_len, cp_size, use_dsa, forward_batch):
    if is_dsa_prefill_cp_round_robin_split():                # round-robin-split 分支
        cur_cp_seq_len = seq_len // cp_size
        assert seq_len % cp_size == 0
    else:                                                    # ★ in-seq-split 分支
        cur_cp_seq_len = seq_len // (cp_size * 2)            # 按 2*cp_size 切（zigzag 两端）
        # Note: (cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
    if (
        cur_cp_seq_len != 0                                  # 切后非空
        and cp_size > 1                                      # CP size > 1
        and use_dsa                                          # DSA 模型
        and forward_batch.forward_mode.is_context_parallel_extend()
        and is_dsa_enable_prefill_cp()
        and sum(forward_batch.extend_seq_lens_cpu) >= cp_size
    ):
        return True
    return False
```

**▶ vs round-robin-split**：in-seq-split 用 `seq_len // (cp_size * 2)` 门槛，要求 `seq_len > 2 * cp_size`；round-robin-split 用 `seq_len // cp_size` 门槛，只要求 `seq_len >= cp_size` 且按 `cp_size` 整除。padding 对齐见 `get_cp_padding_align_size()`（[cp_utils.py:73](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L73)）：in-seq-split 返回 `2 * cp_size`，round-robin-split 返回 `cp_size`。

其中 `ForwardMode.is_context_parallel_extend()`（[forward_batch_info.py:125](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/model_executor/forward_batch_info.py#L125)）对 `EXTEND` 和 `MIXED` 模式返回 `True`（也可选择包含 `DRAFT_EXTEND_V2`）。

> **重要**：对于 `"hello"` 这种只有 1 个 token 的短 prompt，`seq_len=1`，`cur_cp_seq_len = 1 // 16 = 0`，CP 不会真正生效。我们假设 prompt 足够长（比如 64 个 token），这样才能展示完整的 CP 流程。

##### 4b. `prepare_context_parallel_metadata()`（[cp_utils.py:491](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L491)）—— in-seq-split 走完整 zigzag 路径

```python
def prepare_context_parallel_metadata(kv_len, cp_rank, cp_size, seqs_len, extend_seqs_len=None, device="cuda"):
    from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_round_robin_split

    if is_dsa_prefill_cp_round_robin_split():                # round-robin-split 模式
        return ContextParallelMetadata()                     # 返回空元数据（不需要 zigzag 索引重排）

    # ★ in-seq-split 走下面的完整 zigzag 路径
    """prepare_input_dp_with_cp_dsa-zigzag index
    Example (DP_ATTENT_TP == CP_SIZE == 4, single sequence):
        block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
        rank 0: block0, block7
        rank 1: block1, block6
        rank 2: block2, block5
        rank 3: block3, block4
    For bs > 1, each sequence is split into cp_segment_num = 2 * cp_size
    blocks independently; per-rank layout becomes:
        [s0.block_r, s1.block_r, ..., s_{bs-1}.block_r,
         s0.block_{2*cp_size-1-r}, ..., s_{bs-1}.block_{2*cp_size-1-r}]
    i.e. all prev blocks first, then all next blocks -- so torch.split at
    total_q_prev_tokens cleanly separates them.
    """
    assert extend_seqs_len is not None
    # ... padding 对齐，prefix_offsets 计算
    bs = len(extend_seqs_len)
    cp_segment_num = cp_size * 2                              # ★ 段数 = 2 * cp_size
    # ... per_seq_block_sizes, split_list, per_rank_actual_token,
    #     zigzag_index, cp_reverse_index, reverse_split_len,
    #     kv_len_prev_list/next_list, actual_seq_q_prev/next_list ...
    return ContextParallelMetadata(
        split_list=split_list,                               # 各 block 大小（bs * cp_segment_num 段）
        zigzag_index=zigzag_index,                           # 本 rank 持有的 block 索引 [r, 2*cp_size-1-r]
        cp_reverse_index=cp_reverse_index,                   # AllGather 后恢复 block 序的排列
        reverse_split_len=reverse_split_len,                 # AllGather 后各段大小
        per_rank_actual_token=per_rank_actual_token,         # 每 rank 的实际 token 数
        kv_len_prev_tensor=...,                              # prev 段 q 的 KV 右端点（nsa 裸累计，非 nsa 含 prefix）
        kv_len_next_tensor=...,                              # next 段 q 的 KV 右端点
        actual_seq_q_prev_tensor=...,                        # prev 段 q 长度
        actual_seq_q_next_tensor=...,                        # next 段 q 长度
        # ...
    )
```

**in-seq-split 实例**（bs=1, 64 tokens, cp_size=8，`cp_segment_num=16`，每 block = 64/16 = 4 tokens）：

```
split_list        = [4,4,4,...,4]  (16 个 4)                # 16 个 block 的大小
zigzag_index (rank 0) = [0, 15]                             # 本 rank 持有 block0 + block15
cp_reverse_index  = [0,2,4,...,14,15,13,11,...,1]           # AllGather 后恢复 block0..block15 序
per_rank_actual_token = [8,8,...,8]                          # 每 rank 8 tokens（2 block × 4）
kv_len_prev_list (rank 0) = [4]                             # sum blk[:1] = block0（prev 段 q 的 KV 右端点，裸累计）
kv_len_next_list (rank 0) = [64]                            # sum blk[:16] = block0..15（next 段 q 的 KV 右端点，裸累计）
actual_seq_q_prev (rank 0) = [4]                            # block0 的 q 长度
actual_seq_q_next (rank 0) = [4]                            # block15 的 q 长度
```

随后 `dsa_use_prefill_cp()`（[dsa/utils.py:265](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L265)）用于后续各层判断是否启用 CP：

```python
def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):  # 判断当前 forward 是否使用 CP
    if dsa_enable_prefill_cp is None:                        # 若未传入参数，从全局状态读取
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
    if (
        forward_batch.attn_cp_metadata is not None           # CP 元数据已准备好（Step 4 赋值）
        and dsa_enable_prefill_cp                            # 启用了 DSA CP
        and forward_batch.forward_mode.is_context_parallel_extend()  # 是 extend/mixed 模式
    ):
        return True                                          # 三条件全满足 → 使用 CP
    else:
        return False
```

注意 `attn_cp_metadata is not None` 是必要条件——正是上面 `can_dsa_cp_split()` 通过后才会赋值。

**▶ GLM DSA / V2 路径差异**：`GlmMoeDsaForCausalLM` 继承 `DeepseekV2ForCausalLM.forward()`（[deepseek_v2.py:2633](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2633)）。CP 元数据准备逻辑（[deepseek_v2.py:2645-2655](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2645-L2655)）与 V4 类似，但有两点不同：

1. **没有 `apply_cp_reindex()` / `init_flashmla_related()` 调用**：V4 在 round-robin-split 分支中额外执行（[deepseek_v4.py:1740-1748](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1740-L1748)），in-seq-split 与 V2 都不做这些操作。
2. **`use_dsa` 参数来自 `self.use_dsa`**：V2 传入 `can_dsa_cp_split(len_input_ids, self.cp_size, self.use_dsa, forward_batch)`，而 V4 直接传 `True`。

V2 路径的实际代码（[deepseek_v2.py:2645-2655](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2645-L2655)）：

```python
if self.dsa_enable_prefill_cp:                               # 如果启用了 DSA CP（V2 路径）
    if can_dsa_cp_split(                                     # 检查序列是否满足 CP 分割条件
        len_input_ids, self.cp_size, self.use_dsa, forward_batch  # 注意：V2 传 self.use_dsa，V4 传 True
    ):
        forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(  # 构建 CP 元数据
            len_input_ids,                                   # 序列总长度
            self.cp_rank,                                    # 当前 rank
            self.cp_size,                                    # CP 总数
            forward_batch.seq_lens_cpu.tolist(),              # 每个请求的序列长度
            extend_seqs_len=forward_batch.extend_seq_lens_cpu,  # 每个请求的 extend 长度
        )
```

注意 V2 还支持 `mla_enable_prefill_cp`（[deepseek_v2.py:2656-2664](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2656-L2664)），用于非 DSA 的 MLA 模型（如 DeepSeek V3/R1），但 GLM DSA 走的是 `dsa_enable_prefill_cp` 分支。

---

#### Step 5: In-Seq-Split 分发 Token（核心计算 —— zigzag 两端分割）

假设 prompt tokenize 后有 64 个 token：`[t0, t1, t2, ..., t63]`

> **注意**：实际 CP split 操作发生在模型 forward 入口（V4: `DeepseekV4Model.forward`，[deepseek_v4.py:1544](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1544)；V2: `DeepseekV2Model.forward`，[deepseek_v2.py:2337](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2337)），而非 `ForwardBatch.init_new()` 中。所有 rank 先对完整 `input_ids` 做 embedding，然后才执行 split。

##### 5a. Input IDs 的 Round-Robin 分割（仅 V4，a2a 分支）

`cp_round_robin_input_ids()`（[cp_utils.py:191-212](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L191-L212)）有两个分支，in-seq-split 因强制 `moe_a2a_backend = "deepep"` 必走 a2a 分支：

```python
def cp_round_robin_input_ids(input_ids):                     # 对 input_ids 做 round-robin 分割
    cp_size = get_attention_cp_size()                        # 获取 CP 并行度
    cp_rank = get_attention_cp_rank()                        # 获取当前 CP rank
    if get_moe_a2a_backend().is_none():                      # 非 a2a 模式（round-robin-split 默认走此）
        input_ids = input_ids.reshape(-1, cp_size).T.flatten()  # reshape→转置→展平，所有 rank 得到相同的重排序序列
    else:                                                    # ★ a2a 模式（DeepEP，in-seq-split 强制走此）
        input_ids = input_ids[cp_rank::cp_size].contiguous() # 每个 rank 取 stride 切片
    return input_ids
```

**▶ vs round-robin-split**：round-robin-split 默认 `moe_a2a_backend="none"`，走 `reshape(-1, cp_size).T.flatten()` 分支（所有 rank 得到相同重排序的完整 64 token 序列，仅作 routing 索引）。in-seq-split 强制 deepep → 走 `input_ids[cp_rank::cp_size]` stride 切片分支，每 rank 8 tokens：

| CP rank | 拿到的 token IDs |
| ------- | ---------------- |
| 0       | `[t0, t8, t16, t24, t32, t40, t48, t56]` |
| 1       | `[t1, t9, t17, t25, t33, t41, t49, t57]` |
| ...     | ... |
| 7       | `[t7, t15, t23, t31, t39, t47, t55, t63]` |

另外，CP 模式下会设置 `input_ids_global = input_ids`（[deepseek_v4.py:1580](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1580)），该变量传递给每层的 MLP 用于 MoE routing。

**▶ GLM DSA / V2 路径差异**：V2 路径的 `DeepseekV2Model.forward()`（[deepseek_v2.py:2378-2383](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2378-L2383)）**没有 `cp_round_robin_input_ids()` 调用**，也没有 `input_ids_global` 变量。V2 的 `DeepseekV2DecoderLayer.forward()`（[deepseek_v2.py:2054](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2054)）签名中不接收 `input_ids` 或 `input_ids_global` 参数，MLP/MoE 调用（[deepseek_v2.py:2118-2124](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2118-L2124)）也不传递 `input_ids_global`。MoE 的 expert routing 依赖 hidden_states 本身而非 input_ids。

##### 5b. Hidden States 和 Position IDs 的 zigzag 分割

**V4 路径**：在 `DeepseekV4Model.forward()` 中（[deepseek_v4.py:1575-1580](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1575-L1580)）：

```python
if dsa_use_prefill_cp(forward_batch):                        # 如果当前 forward 使用 CP
    if self.pp_group.is_first_rank:                          # 仅 pipeline 并行第一个 rank 做 split
        hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)  # 对 hidden_states 做 zigzag 切片
    positions = cp_split_and_rebuild_position(forward_batch, positions)  # 对 position_ids 做同样的 zigzag 切片
    input_ids = cp_round_robin_input_ids(input_ids)          # 对 input_ids 做 stride 分割（a2a 分支）
    input_ids_global = input_ids                             # 保存分割后的 input_ids 用于 MoE routing
```

`cp_split_and_rebuild_data()`（[cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145)）在 in-seq-split 模式下走 **zigzag 分割路径**：

```python
def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):  # hidden_states 的 CP 分割入口
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():                # round-robin-split 模式
        cp_size = get_attention_cp_size()
        assert input_.shape[0] % cp_size == 0, ...
        return dsa_cp_round_robin_split_data(input_)         # 调用 round-robin stride 切片函数

    # ★ in-seq-split 模式：zigzag 分割
    input_list = list(
        torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)  # 按 split_list 切成 2*cp_size 个 block
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0  # 按 zigzag_index 取 [block_r, block_{2*cp-1-r}] cat
    ).view(-1, input_.shape[-1])
    return result
```

`cp_split_and_rebuild_position()`（[cp_utils.py:167](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L167)）结构类似，in-seq-split 分支按 `zigzag_index` 取 block 并 cat（split 按 `dim=-1`）：

```python
def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():                # round-robin-split 模式
        # ... stride 切片
        return dsa_cp_round_robin_split_data(positions)

    # ★ in-seq-split 模式：zigzag 分割
    position_id_list = list(
        torch.split(positions, forward_batch.attn_cp_metadata.split_list, dim=-1)  # 按 split_list 切
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index],  # 按 zigzag_index 取
        dim=-1,
    )
    return positions
```

对 64 个 token、cp_size=8、`cp_segment_num=16`、每 block=4 tokens 的 3D tensor `[64, hc_mult, hidden_dim]`（V4）或 2D tensor `[64, hidden_dim]`（V2）：

- `torch.split` 按 `split_list=[4,4,...,4]`（16 段）切成 16 个 block
- `zigzag_index = [r, 15-r]`（rank r）→ 取 block_r 和 block_{15-r} cat
- 每 rank 得到 8 tokens（prev=block_r 的 4 tokens + next=block_{15-r} 的 4 tokens）

矩阵变换过程（以 V2 2D tensor `[64, D]` 为例，V4 多一个 hc_mult 维度，原理相同）：

```
原始 hidden_states [64, D]:
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬─────┬─────┬─────┐
│ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │ t6 │ t7 │ t8 │ t9 │ ... │ t62 │ t63 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴─────┴─────┴─────┘

Step 1: 按 split_list=[4,4,...,4] 切成 16 个 block（每 block 4 tokens）

  block0: [t0, t1, t2, t3]        block8:  [t32,t33,t34,t35]
  block1: [t4, t5, t6, t7]        block9:  [t36,t37,t38,t39]
  block2: [t8, t9, t10,t11]       block10: [t40,t41,t42,t43]
  block3: [t12,t13,t14,t15]       block11: [t44,t45,t46,t47]
  block4: [t16,t17,t18,t19]       block12: [t48,t49,t50,t51]
  block5: [t20,t21,t22,t23]       block13: [t52,t53,t54,t55]
  block6: [t24,t25,t26,t27]       block14: [t56,t57,t58,t59]
  block7: [t28,t29,t30,t31]       block15: [t60,t61,t62,t63]

Step 2: 按 zigzag_index=[r, 15-r] 取 [block_r, block_{15-r}] cat

  rank 0: [t0, t1, t2, t3,   t60,t61,t62,t63]   ← block0 + block15 (prev=block0, next=block15)
  rank 1: [t4, t5, t6, t7,   t56,t57,t58,t59]   ← block1 + block14
  rank 2: [t8, t9, t10,t11,  t52,t53,t54,t55]   ← block2 + block13
  rank 3: [t12,t13,t14,t15,  t48,t49,t50,t51]   ← block3 + block12
  rank 4: [t16,t17,t18,t19,  t44,t45,t46,t47]   ← block4 + block11
  rank 5: [t20,t21,t22,t23,  t40,t41,t42,t43]   ← block5 + block10
  rank 6: [t24,t25,t26,t27,  t36,t37,t38,t39]   ← block6 + block9
  rank 7: [t28,t29,t30,t31,  t32,t33,t34,t35]   ← block7 + block8

每 rank 持有 8 tokens：前 4 = prev（block_r），后 4 = next（block_{15-r}）
```

**▶ vs round-robin-split**：round-robin-split 用 `view(-1, cp_size)[:, cp_rank]` 做 stride 切片（rank r 拿 `t_r, t_{r+8}, t_{r+16}, ...`，token 交错）。in-seq-split 持有 zigzag 两端的两整块（连续的 4 + 4 tokens），排列为 `[prev | next]`。这种排列使后续 attention/indexer 可按 prev/next 两段分别处理。

**▶ GLM DSA / V2 路径差异**：V2 的 `DeepseekV2Model.forward()`（[deepseek_v2.py:2378-2383](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2378-L2383)）CP split 代码：

```python
if dsa_use_prefill_cp(                                       # 检查 DSA CP 是否生效
    forward_batch, self.dsa_enable_prefill_cp
) or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp):  # 或 MLA CP 是否生效
    if self.pp_group.is_first_rank:                          # 仅 PP 第一个 rank 做 split
        hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)  # 对 hidden_states 做 zigzag 切片
    positions = cp_split_and_rebuild_position(forward_batch, positions)  # 对 positions 做 zigzag 切片
```

与 V4 对比：只做 `hidden_states` 和 `positions` 的 split，**不调用 `cp_round_robin_input_ids()`**，**不设置 `input_ids_global`**。此外 `dsa_use_prefill_cp` 需要显式传入第二个参数 `self.dsa_enable_prefill_cp`。

---

#### Step 6: Model Forward（每一层的计算过程）

`DeepseekV4ForCausalLM.forward()`（[deepseek_v4.py:1722](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1722)）调用 `self.model.forward()`（[deepseek_v4.py:1751](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1751)），进入 `DeepseekV4Model.forward()`（[deepseek_v4.py:1544](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1544)）。

**单层 Transformer Block 数据流总览**（每个 rank 视角，cp_size=8, 64 tokens）：

```
输入: hidden_states [8, D]（zigzag split 状态：prev 4 tokens | next 4 tokens）

  ┌──────────────────────────────────────────────────────────────────┐
  │                    Self-Attention（prev / next 两段）             │
  │                                                                  │
  │  Q_prev: block_r 的 4 tokens    Q_next: block_{15-r} 的 4 tokens │
  │  KV_prev/next: AllGather+rerange 后完整 64 tokens shape=[64,...] │
  │                                                                  │
  │  ──→ 分 prev/next 两段做 attention（indexer 走 _get_topk_ragged_with_cp ×2）│
  │  ──→ 输出仍是 zigzag split 状态      shape = [8, D]              │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │                  MLP / MoE（强制 DeepEP EP）                     │
  │                                                                  │
  │  ① A2A dispatch（定向分发 token 到 expert 所在 rank）            │
  │  ② MoE per-token 计算（moe_tp_size=1，每卡完整 expert 权重）     │
  │  ③ A2A combine（定向回收）                                       │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓
输出: hidden_states [8, D]（zigzag split 状态，传入下一层）

通信次数: KV AllGather ×1（同步）+ A2A dispatch ×1 + A2A combine ×1 = 1 次同步 + 2×A2A
```

**▶ vs round-robin-split**：round-robin-split 默认 TP MoE，MLP 走 AllGather → AllReduce/ReduceScatter（3 次同步集合通信/层）。in-seq-split 强制 DeepEP EP，MLP 走 A2A dispatch/combine（1 次同步 KV AllGather + 2×A2A/层，可 overlap）。

##### 6a. Embedding 层

```python
hidden_states = self.embed_tokens(input_ids)               # 将 token IDs 映射为 embedding 向量 [64, hidden_dim]
hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)  # mHC 扩展：[64, D] → [64, 1, D] → [64, hc_mult, D]
```

每个 CP rank 上：`hidden_states.shape = [64, hc_mult, hidden_dim]`（所有 rank 相同，因为 embedding 在 split 之前执行）。`unsqueeze` + `repeat` 是 DSV4 特有的 **mHC（multi-Head Composition）扩展**。

接着在 [deepseek_v4.py:1575-1580](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1575-L1580) 执行 CP split（见 Step 5b），分割后每个 rank 上的 `hidden_states.shape` 变为 `[8, hc_mult, hidden_dim]`（zigzag：prev 4 + next 4）。

**▶ GLM DSA / V2 路径差异**：V2 的 `DeepseekV2Model.forward()`（[deepseek_v2.py:2346-2348](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2346-L2348)）embedding 层：

```python
if self.pp_group.is_first_rank:                              # 仅 PP 第一个 rank 做 embedding
    if input_embeds is None:                                 # 若没有传入预计算的 embedding
        hidden_states = self.embed_tokens(input_ids)         # token IDs → embedding 向量 [64, hidden_dim]
```

**没有 `unsqueeze` + `repeat`**——V2 没有 mHC，`hidden_states` 始终是 2D `[seq_len, hidden_dim]`。CP split 后变为 `[8, hidden_dim]`。

##### 6b. 每一层 Transformer Block 的计算

对于每一层，调用 `DeepseekV4DecoderLayer.forward()`（[deepseek_v4.py:1265](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1265)）：

###### i. Self-Attention 层的 `_forward_prepare()`

注意：当 CP 启用时，`MQALayer.__init__`（[deepseek_v4.py:243](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L243)，TP 相关代码于 `:253-259`）将 Attention 层的 TP size 强制设为 1：

```python
self.tp_rank = attn_tp_rank = get_attention_tp_rank()     # 获取 attention TP rank
self.tp_size = attn_tp_size = get_attention_tp_size()     # 获取 attention TP size
self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()   # 是否启用 DSA CP
if self.dsa_enable_prefill_cp:                            # CP 启用时强制 TP=1
    self.cp_size = get_attention_cp_size()                # 保存 CP 并行度
    self.tp_rank = attn_tp_rank = 0                       # 强制 TP rank=0（每 GPU 拥有全部 heads）
    self.tp_size = attn_tp_size = 1                       # 强制 TP size=1（不做 TP 切分）
```

即每个 GPU 拥有全部的 attention heads（权重复制），独立计算完整的 attention。两模式一致。

`MQALayer._forward_prepare()`（[deepseek_v4.py:698](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L698)）是 CP 最关键的部分。实际代码有多条分支路径（`fuse_wqa_wkv`、`use_fused_qk_norm_rope` 等），以下为 CP 相关核心逻辑：

```python
# 1. 计算 Q（每个 rank 独立计算自己的 Q，含 prev 4 + next 4 = 8 tokens）
use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)  # 判断当前层是否使用 CP

# 以非 fused_qk_norm_rope 路径为例（line 774-796）：
q_lora = self.q_norm(q_lora)                            # 对 Q 的 LoRA 输出做 norm
q = self._compute_q_b(q_lora, positions, q_out)         # Q 的 up projection + RoPE 旋转位置编码

# 2. 计算 KV（关键！需要 all-gather）
if use_cp:                                               # CP 路径
    kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)  # 计算本 rank 的 KV（bf16 格式用于通信）
    # ★★★ 核心 all-gather ★★★
    kv = cp_all_gather_rerange_output(                   # 从所有 rank 收集 KV 并恢复 token 顺序
        kv.contiguous(), self.cp_size, forward_batch,    # 传入 KV tensor、CP 大小、batch 信息
        torch.cuda.current_stream(),                     # 当前 CUDA stream
    )
    # 此时 kv shape: [64, hc_mult, kv_dim] — 收集了所有 8 个 rank 的 KV！
    attn_backend.store_cache(                            # 将完整 KV 写入 attention cache
        layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch,  # 指定层 ID 和 forward_batch
    )
else:                                                    # 非 CP 路径
    self._compute_kv_to_cache(x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a)  # 直接计算 KV 并写入 cache
```

`_compute_kv_bf16()`（[deepseek_v4.py:499](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L499)）为 CP 专用路径，生成 bf16 格式的 KV 用于跨 rank all-gather，而非直接写入 cache。

**▶ GLM DSA / V2 路径差异**：V2 的 Attention 层为 `DeepseekV2AttentionMLA`（[deepseek_v2.py:1425](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1425)），CP 初始化（[deepseek_v2.py:1469-1477](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1469-L1477)）：

```python
self.dsa_enable_prefill_cp = dsa_enable_prefill_cp              # 保存 DSA CP 启用标志
self.mla_enable_prefill_cp = mla_enable_prefill_cp              # 保存 MLA CP 启用标志
if self.dsa_enable_prefill_cp:                                  # DSA CP 要求模型必须使用 DSA
    assert self.use_dsa, "CP currently only supports deepseek v3.2 model"  # 断言检查
if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:    # 任意一种 CP 启用时
    self.cp_size = get_attention_cp_size()                      # 保存 CP 并行度
```

注意 V2 **不强制 TP=1**（没有 V4 的 `self.tp_rank = 0; self.tp_size = 1`）。但 CP 复用了 attn_tp comm group，实际 `get_attention_tp_size()` 在 CP 启用时返回 1（由 `_handle_context_parallelism` 中 `tp_size % attn_cp_size == 0` 保证）。

V2 的 KV all-gather 通过 `rebuild_cp_kv_cache()`（[deepseek_v2.py:1874](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1874)）实现，直接操作 MLA 的 latent cache：

```python
def rebuild_cp_kv_cache(self, latent_cache, forward_batch, k_nope, k_pe):  # V2 KV latent cache all-gather
    latent_cache[..., : self.kv_lora_rank] = k_nope.squeeze(1)  # 将 k_nope 写入 latent cache 前半部分
    latent_cache[..., self.kv_lora_rank :] = k_pe.squeeze(1)    # 将 k_pe 写入 latent cache 后半部分
    latent_cache_output = cp_all_gather_rerange_output(         # 从所有 rank 收集 latent cache 并恢复顺序
        latent_cache.contiguous(),                              # 输入：本 rank 的 latent cache
        self.cp_size,                                           # CP 并行度
        forward_batch,                                          # forward_batch 信息
        torch.cuda.current_stream(),                            # 当前 CUDA stream
    )
    k_nope = latent_cache_output[..., : self.kv_lora_rank].unsqueeze(1)  # 从 all-gather 结果中取回完整 k_nope
    k_pe = latent_cache_output[..., self.kv_lora_rank :].unsqueeze(1)    # 从 all-gather 结果中取回完整 k_pe
    return k_nope, k_pe                                         # 返回完整的 KV（所有 64 个 token）
```

与 V4 的区别：V4 all-gather 整个 KV tensor（含 mHC 维度），V2 all-gather 的是拼接的 `[k_nope, k_pe]` latent cache（MLA 压缩表示），all-gather 后再拆回 `k_nope` 和 `k_pe`。底层都调用同一个 `cp_all_gather_rerange_output()`。

###### ii. `cp_all_gather_rerange_output()` 的计算细节（in-seq-split 走 cp_reverse_index 复杂 rerange）

`cp_all_gather_rerange_output()`（[cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310)）执行流程：

```python
def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):  # AllGather + rerange 恢复顺序
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():                # round-robin-split 模式（简单 transpose rerange）
        # ... view(cp_size, -1).transpose(0,1).reshape ... (见 round-robin-split 文档)
        return output_tensor

    # ★ in-seq-split 模式：cp_reverse_index 复杂 rerange
    # TODO: Do we need to remove the padding here?
    bs_seq_len, hidden_size = input_tensor.shape
    output_tensor = cp_all_gather_reorganized_into_tensor(   # AllGather + pad/remove（rerange 前置）
        input_tensor, cp_size, forward_batch, stream,
    )
    outputs_list = list(
        torch.split(                                          # 按 reverse_split_len 切段
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],  # 按 cp_reverse_index 重排段
        dim=0,
    )
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor
```

`cp_all_gather_reorganized_into_tensor()`（[cp_utils.py:215](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L215)）负责 AllGather + padding 对齐 + 按每 rank 实际 token 数截取：

```python
def cp_all_gather_reorganized_into_tensor(input_tensor, cp_size, forward_batch, stream):
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]  # 所有 rank 中最长的 token 数
    pad_size = max_len - input_tensor.shape[0]               # 本 rank 不足 max_len 则 pad
    if pad_size > 0:
        input_tensor = F.pad(input_tensor, (0, 0, 0, pad_size), mode="constant", value=0)
    with use_symmetric_memory(...):
        input_tensor_full = torch.empty(max_len * cp_size, *input_tensor.shape[1:], ...)  # 预分配接收 buffer
    get_attention_cp_group().cp_all_gather_into_tensor_async(input_tensor_full, input_tensor, stream)  # NCCL AllGather
    outputs_list_max = list(torch.split(input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0))  # 按 max_len 切 cp_size 段
    outputs = torch.cat(                                     # 按 per_rank_actual_token 截取实际 token（去 pad）
        [outputs_list_max[index][:per_rank_len]
         for index, per_rank_len in enumerate(forward_batch.attn_cp_metadata.per_rank_actual_token)],
        dim=0,
    )
    return outputs
```

**in-seq-split 实例：KV 的 AllGather + Rerange 过程**（bs=1, 64 tokens, cp_size=8, 每 block=4 tokens）：

```
① 各 rank 的 KV（zigzag split 状态，每 rank 8 tokens = prev 4 + next 4）：

  rank 0: KV[t0,t1,t2,t3, t60,t61,t62,t63]    (prev=block0       next=block15)
  rank 1: KV[t4,t5,t6,t7, t56,t57,t58,t59]    (prev=block1       next=block14)
  rank 2: KV[t8,t9,t10,t11,t52,t53,t54,t55]   (prev=block2       next=block13)
  rank 3: KV[t12,t13,t14,t15,t48,t49,t50,t51] (prev=block3       next=block12)
  rank 4: KV[t16,t17,t18,t19,t44,t45,t46,t47] (prev=block4       next=block11)
  rank 5: KV[t20,t21,t22,t23,t40,t41,t42,t43] (prev=block5       next=block10)
  rank 6: KV[t24,t25,t26,t27,t36,t37,t38,t39] (prev=block6       next=block9)
  rank 7: KV[t28,t29,t30,t31,t32,t33,t34,t35] (prev=block7       next=block8)

② AllGather 按 rank 顺序拼接（每 rank 内 prev 在前、next 在后）：

  KV[t0..t3,t60..t63, t4..t7,t56..t59, ..., t28..t31,t32..t35]   shape=[64, kv_dim]
   └── rank0 ──────┘  └── rank1 ──────┘      └── rank7 ──────┘

③ split by reverse_split_len=[4,4,4,4,...,4] (16 段，每 rank [prev 4, next 4]) → 16 段。
   post-AG 顺序为 [rank0_prev, rank0_next, rank1_prev, rank1_next, ...]，即偶数 seg = block_r (prev)，奇数 seg = block_{15-r} (next)：

  seg0  = block0  = [t0,t1,t2,t3]       seg8  = block4  = [t16,t17,t18,t19]
  seg1  = block15 = [t60,t61,t62,t63]   seg9  = block11 = [t44,t45,t46,t47]
  seg2  = block1  = [t4,t5,t6,t7]       seg10 = block5  = [t20,t21,t22,t23]
  seg3  = block14 = [t56,t57,t58,t59]   seg11 = block10 = [t40,t41,t42,t43]
  seg4  = block2  = [t8,t9,t10,t11]     seg12 = block6  = [t24,t25,t26,t27]
  seg5  = block13 = [t52,t53,t54,t55]   seg13 = block9  = [t36,t37,t38,t39]
  seg6  = block3  = [t12,t13,t14,t15]   seg14 = block7  = [t28,t29,t30,t31]
  seg7  = block12 = [t48,t49,t50,t51]   seg15 = block8  = [t32,t33,t34,t35]

④ 按 cp_reverse_index 重排段顺序（bs=1 时为 range(0,16,2)+range(15,0,-2)）
   = [0,2,4,6,8,10,12,14, 15,13,11,9,7,5,3,1]
   → seg[0],seg[2],seg[4],seg[6],seg[8],seg[10],seg[12],seg[14],
     seg[15],seg[13],seg[11],seg[9],seg[7],seg[5],seg[3],seg[1]
   = block0,block1,block2,block3,block4,block5,block6,block7,
     block8,block9,block10,block11,block12,block13,block14,block15

  KV[t0..t3, t4..t7, t8..t11, ..., t60..t63]
  ✓ 恢复为原始 block 顺序（block0..block15 = t0..t63）！
```

通信开销：NCCL AllGather，每个 rank 发送 `8 * hc_mult * kv_dim` 数据，总共接收 `64 * hc_mult * kv_dim`。

**▶ vs round-robin-split**：round-robin-split 的 rerange 是简单的 `view(cp_size, -1).transpose(0,1).reshape`（3 步矩阵变换），因为 stride 切片有规整的交错结构。in-seq-split 因为 zigzag 两端排列（每 rank [prev, next] 不连续），需要 `cp_reverse_index` 显式重排段顺序，逻辑复杂。

###### iii. DSA Indexer（in-seq-split 走 prev/next 分段 topk）

```python
if self.indexer is not None:                                 # 如果配置了 DSA Indexer（C4Indexer 等）
    self.indexer(x, q_lora, forward_batch, attn_backend)     # 执行稀疏 attention 的索引计算
```

in-seq-split 模式下，DSA Indexer（[dsa_indexer.py:1484-1528](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1484-L1528)）把本 rank 的 Q 按 prev/next 分两段分别做 topk：

```python
if (
    forward_batch.attn_cp_metadata is not None
    and is_dsa_prefill_cp_in_seq_split()                     # ★ in-seq-split 分段分支
):
    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev_list[0]      # prev 段 KV 长度
    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next_list[0]      # next 段 KV 长度
    actual_seq_q_prev = forward_batch.attn_cp_metadata.actual_seq_q_prev_list[0]  # prev 段 q 长度
    actual_seq_q_next = forward_batch.attn_cp_metadata.actual_seq_q_next_list[0]  # next 段 q 长度

    # TODO support mutil-batch                                   # ★ 多 batch 支持有精度问题，故强制 batch_size=1
    q_fp8_prev, q_fp8_next = torch.split(                     # Q 按 (n+1)//2 分 prev/next 两段
        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
    )
    weights_prev, weights_next = torch.split(
        weights, (weights.shape[0] + 1) // 2, dim=0
    )
    topk_result_prev = self._get_topk_ragged_with_cp(         # prev 段 topk
        forward_batch, layer_id, q_fp8_prev, weights_prev, metadata,
        kv_len_prev, actual_seq_q_prev,
    )
    topk_result_next = self._get_topk_ragged_with_cp(         # next 段 topk
        forward_batch, layer_id, q_fp8_next, weights_next, metadata,
        kv_len_next, actual_seq_q_next,
    )
    topk_result = torch.cat([topk_result_prev, topk_result_next], dim=0)  # 拼回
    topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
    return maybe_capture_indexer_topk(layer_id, topk_result)
elif is_in_piecewise_cuda_graph():                           # piecewise CUDA graph 路径（in-seq-split 禁用）
    # ...
else:                                                        # 非 CP 正常 ragged 路径
    topk_result = self._get_topk_ragged(...)
```

`_get_topk_ragged_with_cp()`（[dsa_indexer.py:960](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L960)）按 `kv_len` / `actual_seq_q` 对 prev 或 next 段独立做稀疏 topk 索引，断言禁止 piecewise CUDA graph（[dsa_indexer.py:971-973](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L971-L973)，对应 Step 0 的 `disable_piecewise_cuda_graph = True`）。

**▶ vs round-robin-split**：round-robin-split **不走**此 prev/next 分段分支，走正常 `_get_topk_ragged` 路径（或 piecewise CUDA graph 路径），无需 `kv_len_prev/next`、`actual_seq_q_prev/next` 元数据。这也是 in-seq-split 必须生成完整 `ContextParallelMetadata`（含 prev/next 字段）而 round-robin-split 返回空的原因。

###### iv. Attention Forward（DSA Backend）

```python
attn_output = attn_backend.forward(q, ...)                   # Q=[8,heads,d] 对 KV=[64,...] 做 attention 计算
```

> **关键**：Q 只有 1/8，但 KV 是完整的！每个 rank 计算一部分 Q 的 attention，但需要看到所有 KV。由于 TP=1，每个 GPU 独立完成 attention，无需 TP 通信。

Attention 输出后，执行 DSV4 特有的 **inverse RoPE** 操作。

###### v. Attention 输出保持 split 状态

Attention output 保持 split 状态（每个 rank 只有 8 个 token 的结果，仍为 `[prev 4 | next 4]` zigzag 排列），不立即 all-gather。

###### vi. MLP / MoE 层（强制 DeepEP A2A，不走 AllGather/ReduceScatter）

MLP 层的处理在 `DeepseekV4DecoderLayer.forward()` 中（[deepseek_v4.py:1387-1428](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1387-L1428)）：

```python
_use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)  # 判断本层是否使用 CP

if _use_cp:                                                  # CP 启用时
    if get_moe_a2a_backend().is_none():                      # 非 a2a 模式（round-robin-split 默认走此）
        hidden_states = dsa_cp_gather_hidden_states(hidden_states)  # AllGather: [8,D] → [64,D]（无 rerange）
    else:                                                    # ★ a2a 模式（DeepEP，in-seq-split 强制走此）
        assert get_moe_a2a_backend().is_deepep(), (           # 断言必须是 DeepEP 后端
            "CP requires DeepEP (moe_a2a_backend == deepep). "
            "Only DeepEP is tested with CP's per-rank token split."
        )
        # 不做 AllGather，直接进 MLP，由 DeepEP 内部 dispatch 处理跨 rank token 流动

hidden_states = self.mlp(hidden_states, ..., use_reduce_scatter=_use_cp)  # MLP/MoE 前向计算

if _use_cp and get_moe_a2a_backend().is_none():              # CP + 非 a2a 模式才做 ReduceScatter
    hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)  # ReduceScatter: [64,D] → [8,D]
# in-seq-split（deepep）不走此分支，由 DeepEP combine 负责回收
```

**in-seq-split MLP 全流程（DeepEP EP）**：

```
┌─── Attn 输出（zigzag split 状态）────────────────────────────────────┐
│ rank 0: [t0,t1,t2,t3, t60,t61,t62,t63]   shape=[8, D]              │
│ rank 1: [t4,t5,t6,t7, t56,t57,t58,t59]                             │
│  ...                                                                │
│ rank 7: [t28,t29,t30,t31, t32,t33,t34,t35]                         │
└────────────────────────────────────────────────────────────────────┘
                             ↓ 不做 AllGather，直接进 DeepEP dispatch
┌─── A2A dispatch：按 routing 把每个 token 发到持有目标 expert 的 rank ┐
│   (dispatch 后 token 顺序与 expert 分布有关，不再是原 rank 的 8 tokens)│
└────────────────────────────────────────────────────────────────────┘
                             ↓ MoE 计算（moe_tp_size=1，每卡 1/cp_size 个完整 expert）
┌─── MoE per-token 计算：无 TP partial、无 AllReduce ─────────────────┐
│   每卡持有 1/cp_size 个完整 expert（EP 模式），权重不切分            │
└────────────────────────────────────────────────────────────────────┘
                             ↓ A2A combine：把 expert 输出按原 token 归还到原 rank
┌─── 回到 zigzag split 状态 ──────────────────────────────────────────┐
│ rank 0: [r0,r1,r2,r3, r60,r61,r62,r63]   shape=[8, D]              │
│ rank 1: [r4,r5,r6,r7, r56,r57,r58,r59]                             │
│  ...                                                                │
│ rank 7: [r28,r29,r30,r31, r32,r33,r34,r35]                         │
└────────────────────────────────────────────────────────────────────┘
                             ↓ 传入下一层
```

**为什么 in-seq-split 不用 `use_reduce_scatter=True` 触发 ReduceScatter？**  
`use_reduce_scatter=_use_cp` 仍传 `True`，使 `should_skip_post_experts_all_reduce()`（[moe/utils.py:422](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L422)）返回 `True`（line 446-447），跳过 MoE 内部的 post-experts AllReduce：

```python
def should_skip_post_experts_all_reduce(           # 判断是否跳过 MoE 内部的 post-experts all-reduce
    *, is_tp_path: bool,                           # 是否走 TP 路径
    use_reduce_scatter: bool = False,              # 是否使用 ReduceScatter（CP 时为 True）
    should_allreduce_fusion: bool = False,         # 是否使用 allreduce fusion
) -> bool:
    if should_allreduce_fusion or use_reduce_scatter:  # CP 时 use_reduce_scatter=True → 跳过内部 all-reduce
        return True
    # ...
    return False
```

但 in-seq-split 下 `moe_tp_size = 1`（EP 模式），本就没有 TP partial result 需要合并，MoE 内部本就不会触发 TP AllReduce。`use_reduce_scatter=True` 主要影响 round-robin-split 的 TP MoE 路径（跳过 AllReduce 改由 ReduceScatter 替代）。in-seq-split 下 DeepEP 的 dispatch/combine 已自带跨 rank token 流动，不需要额外的 AllGather/ReduceScatter。

**▶ vs round-robin-split**：round-robin-split 默认 TP MoE（`ep_size=1`、`moe_tp_size=tp_size`），MLP 走 `dsa_cp_gather_hidden_states`（AllGather，无 rerange）→ MoE（TP partial）→ `dsa_cp_reduce_scatter_hidden_states`（ReduceScatter，同时完成 TP reduce + CP scatter），每层 3 次同步集合通信。可选 `--moe-a2a-backend deepep --ep-size <tp_size>` 切到 DeepEP 路径。in-seq-split 强制 DeepEP，无 TP MoE 选项。

**▶ GLM DSA / V2 路径差异（完整描述）**：V2/GLM 的 MLP 通信通过 `DSACPLayerCommunicator` 自动管理，V4 显式调用。V2 的 `DeepseekV2DecoderLayer`（[deepseek_v2.py:1900](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1900)）在 CP 启用时使用 `DSACPLayerCommunicator`（[deepseek_v2.py:2007-2020](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2007-L2020)）：

```python
if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:  # 任意 CP 启用时创建 communicator
    # DSACPLayerCommunicator is flavor-agnostic; its internal gates
    # read both dsa_use_prefill_cp and mla_use_prefill_cp. The rename
    # to CPLayerCommunicator is deferred to a cleanup PR.
    self.layer_communicator = DSACPLayerCommunicator(            # 创建层间通信管理器
        layer_scatter_modes=self.layer_scatter_modes,           # 每层的 scatter 模式配置
        input_layernorm=self.input_layernorm,                   # 输入 LayerNorm（gather 后可能需要）
        post_attention_layernorm=self.post_attention_layernorm, # attention 后的 LayerNorm
        allow_reduce_scatter=True,                              # 允许使用 ReduceScatter
        is_last_layer=(...),                                    # 是否为最后一层
        qkv_latent_func=self.self_attn.prepare_qkv_latent,     # QKV latent 准备函数引用
    )
```

V2 的 `DeepseekV2DecoderLayer.forward()`（[deepseek_v2.py:2054](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2054)）通过 communicator 隐式处理 MLP 通信：

```python
hidden_states, residual = self.layer_communicator.prepare_mlp(      # MLP 前处理（CP + 非 a2a 时 AllGather + LayerNorm）
    hidden_states, residual, forward_batch                          # in-seq-split（deepep）下 _compute_mlp_mode 返回 SCATTERED，不 gather
)
use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(  # 判断是否使用 ReduceScatter
    forward_batch                                                   # CP 启用时返回 True
)
hidden_states = self.mlp(                                           # MLP/MoE 前向计算
    hidden_states, forward_batch,                                   # 输入 hidden_states
    should_allreduce_fusion, use_reduce_scatter,                    # use_reduce_scatter=True 跳过内部 all-reduce
    gemm_output_zero_allocator,                                     # GEMM 输出的零值分配器
)
hidden_states, residual = self.layer_communicator.postprocess_layer(  # MLP 后处理（CP + 非 a2a 时 ReduceScatter）
    hidden_states, residual, forward_batch                          # in-seq-split（deepep）下不 scatter
)
```

关键区别：
1. V2 的 gather/scatter 由 `DSACPLayerCommunicator` 的 `prepare_mlp()` 和 `postprocess_layer()` 内部处理，而非 V4 的显式 `dsa_cp_gather_hidden_states()` / `dsa_cp_reduce_scatter_hidden_states()` 调用。in-seq-split 下因 `moe_a2a_backend=deepep`，`_compute_mlp_mode()`（[communicator.py:378](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator.py#L378)，SCATTERED 返回于 `:380-382`）返回 `ScatterMode.SCATTERED`，`prepare_mlp`/`postprocess_layer` 实际不触发 gather/scatter（由 DeepEP a2a 处理）。
2. V2 的 MLP 调用（[deepseek_v2.py:2118-2124](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2118-L2124)）**不传递 `input_ids` 或 `input_ids_global`**——V4 的 `self.mlp(hidden_states, ..., input_ids=input_ids, input_ids_global=input_ids_global, use_reduce_scatter=_use_cp)` 在 V2 中简化为 `self.mlp(hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter, ...)`。
3. V2 有独立的 `residual` tensor 贯穿整个 decoder layer（标准 residual connection），V4 使用 `prev_residual` / `prev_post` / `prev_comb`（mHC 特有）。

V2/GLM 的 MoE 层（[deepseek_v2.py:1029](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1029)、[glm4_moe.py:645](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/glm4_moe.py#L645)）有相同的 `should_skip_post_experts_all_reduce` 判断：

```python
# deepseek_v2.py:1029-1034 (DeepseekV2MoE.forward_normal) — 正常 TP AllReduce
if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # in-seq-split 下 moe_tp_size=1，此条件不成立
    is_tp_path=True,
    use_reduce_scatter=use_reduce_scatter,         # use_reduce_scatter=True → should_skip 返回 True → 跳过
    should_allreduce_fusion=should_allreduce_fusion,
):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 合并各 rank 的 TP partial result
```

in-seq-split 下 `moe_tp_size = 1`，`self.tp_size > 1` 不成立，本就不做 AllReduce。

###### vii. 进入下一层

经过 MLP 后，`hidden_states` 再次变成 zigzag split 状态，每个 rank 拥有对应的 prev+next 两整块（如 rank 0: `[t0,t1,t2,t3, t60,t61,t62,t63]`）。下一层重复上述过程。

---

#### Step 7: 最后一层 → Logits → Sampling

经过所有 Transformer 层后，只在最后一层之后做一次完整的恢复（`DeepseekV4Model.forward`，[deepseek_v4.py:1615-1621](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1615-L1621)）：

```python
if self.pp_group.is_last_rank and dsa_use_prefill_cp(forward_batch):  # PP 最后一个 rank + CP 启用
    hidden_states = cp_all_gather_rerange_output(            # 最终 AllGather + rerange 恢复完整序列
        hidden_states, self.cp_size, forward_batch,          # [8,hc,D] × 8 ranks → [64,hc,D]（正序）
        torch.cuda.current_stream(),                         # 当前 CUDA stream
    )
```

此时 `hidden_states` 恢复为完整序列 `[64, hc_mult, hidden_dim]`（token 顺序正确，经 `cp_reverse_index` rerange 恢复）。

```
最终恢复流程（V4 路径）:

  rank 0          rank 1                 rank 7
  [8,hc,D]       [8,hc,D]     ...      [8,hc,D]
  t0..t3,t60..63 t4..t7,t56..59         t28..31,t32..35
       │               │                     │
       └───────── AllGather ──────────────────┘
                       ↓
              [64, hc_mult, hidden_dim]  (按 rank 拼接，每 rank [prev|next])
                       ↓ split by reverse_split_len + cp_reverse_index 重排
              [64, hc_mult, hidden_dim]  (正序: t0,t1,...,t63) ✓
                       ↓
                  flatten(1)  →  [64, hc_mult × hidden_dim]
                       ↓
                   hc_head()  →  [64, hidden_dim]          ← mHC 压缩
                       ↓
                    norm()    →  [64, hidden_dim]
                       ↓
               logits_processor  →  采样

V2 路径差异: norm() → AllGather+rerange → logits（无 hc_head，2D 始终）
```

随后合并 `hc_mult` 维度并归一化（[deepseek_v4.py:1627-1632](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1627-L1632)）：

```python
pre_hc_head = hidden_states.flatten(1)                    # 展平 mHC 维度 [64, hc_mult, D] → [64, hc_mult*D]
hidden_states = self.hc_head(                             # mHC 压缩：[64, hc_mult*D] → [64, D]
    hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base  # 传入压缩函数和缩放参数
)
hidden_states = self.norm(hidden_states)                  # RMSNorm 归一化 [64, D] → [64, D]
```

`DeepseekV4Model.forward()` 返回 `(hidden_states, pre_hc_head)`（[deepseek_v4.py:1634](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1634)）。

回到 `DeepseekV4ForCausalLM.forward()`（[deepseek_v4.py:1760-1768](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1760-L1768)）：

```python
hidden_states, pre_hc_head = hidden_states                # 解包 Model.forward() 的返回值
return self.logits_processor(                             # LogitsProcessor 处理最终输出
    input_ids, hidden_states, self.lm_head, forward_batch,  # token IDs、hidden states、LM head、batch 信息
    aux_hidden_states,                                    # 辅助 hidden states（若有）
    hidden_states_before_norm=pre_hc_head,                # norm 前的 hidden states（V4 mHC 特有）
)
```

其中 `self.lm_head` 是 `ParallelLMHead`（初始化于 [deepseek_v4.py:1657](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1657)），`self.logits_processor` 是 `LogitsProcessor`（初始化于 [deepseek_v4.py:1666](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1666)）。`LogitsProcessor` 负责：对 `hidden_states` 投影到 vocab 维度、提取需要采样的 token 位置的 logits、应用 `sampling_params` 等。

**▶ GLM DSA / V2 路径差异**：V2 的最后阶段与 V4 有两个关键区别。

**区别 1：Norm 在 all-gather 之前**。V2 的 `DeepseekV2Model.forward()`（[deepseek_v2.py:2458-2475](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2458-L2475)）：

```python
# 先 norm（每个 rank 独立对自己的 split 做 norm）
else:                                                    # 非 idle 模式
    if not forward_batch.forward_mode.is_idle():         # 再次检查确认非 idle
        if residual is None:                             # 如果没有 residual
            hidden_states = self.norm(hidden_states)            # 直接 norm [8,D] → [8,D]
        else:                                            # 有 residual（标准 residual connection）
            hidden_states, _ = self.norm(hidden_states, residual)  # residual fusion norm

# 后 all-gather（norm 后的结果再收集）
if self.pp_group.is_last_rank and (                     # PP 最后一个 rank 且
    dsa_use_prefill_cp(forward_batch, self.dsa_enable_prefill_cp)  # DSA CP 启用
    or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp)  # 或 MLA CP 启用
):
    # allgather + rerrange
    hidden_states = cp_all_gather_rerange_output(               # AllGather + rerange 恢复完整序列
        hidden_states, self.cp_size, forward_batch,          # [8,D] × 8 ranks → [64,D]（正序）
        torch.cuda.current_stream(),                         # 当前 CUDA stream
    )
```

V4 的顺序是：all-gather → hc_head → norm。V2 的顺序是：norm → all-gather。由于 RMSNorm 是 per-token 操作，两者数学上等价，但 V2 在 all-gather 时传输的是 norm 后的 `hidden_dim` 大小数据，V4 传输的是 norm 前的 `hc_mult * hidden_dim` 大小数据（mHC 尚未压缩）。

**区别 2：Logits 调用更简单**。V2 的 `DeepseekV2ForCausalLM.forward()`（[deepseek_v2.py:2674-2677](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2674-L2677)）：

```python
if self.pp_group.is_last_rank:                          # 仅 PP 最后一个 rank 计算 logits
    return self.logits_processor(                        # LogitsProcessor 处理最终输出
        input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states  # V2 无 pre_hc_head
    )
```

无 `hidden_states_before_norm=pre_hc_head` 参数（V4 用它传递 norm 前的 hidden states 给 `LogitsProcessor`），因为 V2 没有 mHC，也不需要 `pre_hc_head`。

---

#### Step 8: 结果回传

1. `TpModelWorker` 返回 logits 经采样后得到 `next_token_ids`
2. Scheduler 拿到 `next_token_id`，追加到 `req.output_ids`
3. Scheduler 构造 `BatchStrOutput` / `BatchTokenIDOutput`，通过 ZMQ 发给 DetokenizerManager
4. Detokenizer 解码 token_id → text
5. 结果通过 ZMQ 回到 TokenizerManager
6. TokenizerManager 通过 HTTP response 返回给用户

---

#### Step 9: Decode 阶段（后续 token 生成）

Prefill 完成后，后续每个 decode step：

- 不再使用 CP（`ForwardMode.DECODE`，`is_context_parallel_extend()`（[forward_batch_info.py:125](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/model_executor/forward_batch_info.py#L125)）对 `DECODE` 返回 `False`）：

  ```python
  def is_context_parallel_extend(self, include_draft_extend_v2: bool = False):  # 判断是否为 CP extend 模式
      return (                                                # 满足以下任意一种模式
          self == ForwardMode.EXTEND                          # EXTEND 模式（纯 prefill）
          or self == ForwardMode.MIXED                        # MIXED 模式（prefill + decode 混合）
          or (                                                # 或者
              self == ForwardMode.DRAFT_EXTEND_V2             # DRAFT_EXTEND_V2 模式（投机解码）
              if include_draft_extend_v2                      # 若显式传入 include_draft_extend_v2=True
              else False                                      # 否则不包含 DRAFT_EXTEND_V2
          )
      )
  ```

  Decode 阶段 `ForwardMode.DECODE` 不匹配以上任何条件，因此 `dsa_use_prefill_cp()` 返回 `False`，CP 不生效。两模式行为一致。

- 每个 rank 独立处理 1 个 token 的 attention（使用完整的 KV cache）
- 直到生成 EOS 或达到 `max_new_tokens`

---

#### 附录 A：关键函数索引

| 函数名 | 文件位置 | 作用 |
|--------|----------|------|
| `_handle_context_parallelism` | [server_args.py:3218](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3218) (调用于 `:1000`) | 参数校验 |
| `_handle_model_specific_adjustments` | [server_args.py:1792](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1792) (DSA-CP 子块 `:1857-1893`) | in-seq-split 强制 deepep+ep=tp 的自动配置 |
| `is_dsa_prefill_cp_in_seq_split` | [dsa/utils.py:71](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L71) | 判断是否为 in-seq-split 模式 |
| `is_prefill_cp_in_seq_split` | [cp_utils.py:66](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L66) | 非 DSA 版（普通 MLA CP）同义判断 |
| `get_cp_padding_align_size` | [cp_utils.py:73](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L73) | in-seq-split 返回 `2*cp_size`，round-robin 返回 `cp_size` |
| `can_dsa_cp_split` | [dsa/utils.py:175](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L175) | in-seq-split 门槛 `seq_len // (cp_size*2)` |
| `prepare_context_parallel_metadata` | [cp_utils.py:491](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L491) | 生成 zigzag 索引（in-seq-split）；round-robin 返回空 |
| `cp_split_and_rebuild_data` | [cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145) | 按 zigzag_index 切 block 并 cat（in-seq-split） |
| `cp_split_and_rebuild_position` | [cp_utils.py:167](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L167) | position ids 同理 zigzag split |
| `cp_round_robin_input_ids` | [cp_utils.py:191](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L191) | input_ids stride/transpose（in-seq-split 走 a2a stride 分支） |
| `cp_all_gather_reorganized_into_tensor` | [cp_utils.py:215](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L215) | AllGather + pad/remove（in-seq-split rerange 前置） |
| `cp_all_gather_rerange_output` | [cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310) | AllGather + `cp_reverse_index` 恢复 block 序 |
| `cp_all_gather_rerange_kv_cache` | [cp_utils.py:381](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L381) | KV cache 版 rerange |
| `dsa_use_prefill_cp` | [dsa/utils.py:265](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L265) | 判断当前 forward 是否使用 CP |
| `dsa_cp_gather_hidden_states` | [communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55) | MLP 前 AllGather（TP 模式，in-seq-split 不走） |
| `dsa_cp_reduce_scatter_hidden_states` | [communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67) | MLP 后 ReduceScatter（TP 模式） |
| `DSACPLayerCommunicator` | [communicator_dsa_cp.py:79](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L79) | V2/GLM CP 层间通信管理器 |
| `should_skip_post_experts_all_reduce` | [moe/utils.py:422](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L422) | CP 下跳过 MoE 内部 all-reduce |
| `MQALayer.__init__` | [deepseek_v4.py:243](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L243) | CP 下 TP 强制为 1（`:253-259`） |
| `MQALayer._forward_prepare` | [deepseek_v4.py:698](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L698) | Q/KV 计算 + KV all-gather |
| `_get_topk_ragged_with_cp` | [dsa_indexer.py:960](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L960) | in-seq-split prev/next 分段 topk |
| `_get_topk_ragged` (CP 分支) | [dsa_indexer.py:1484-1528](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1484-L1528) | in-seq-split prev/next 分段入口 |
| `DeepseekV4Model.forward` | [deepseek_v4.py:1544](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1544) | 模型 forward（含 CP split/gather） |
| `DeepseekV4ForCausalLM.forward` | [deepseek_v4.py:1722](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1722) | 顶层 forward（CP 元数据准备 + logits） |
| `DeepseekV4DecoderLayer.forward` | [deepseek_v4.py:1265](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1265) | 单层 forward（MLP A2A 路径，`:1387-1428`） |
| `PrefillAdder.add_one_req` | [schedule_policy.py:845-861](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/managers/schedule_policy.py#L845-L861) | in-seq-split 强制 batch_size=1 |
| `get_attention_cp_rank` | [dp_attention.py:338](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L338) | 获取当前 CP rank |
| `get_attention_cp_size` | [dp_attention.py:342](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L342) | 获取 CP size |
| `attn_cp_all_gather_into_tensor` | [dp_attention.py:599](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L599) | 对 CP group 执行 AllGather |
| `DeepseekV2ForCausalLM.forward` | [deepseek_v2.py:2633](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2633) | V2 顶层 forward（GLM DSA 继承） |
| `DeepseekV2Model.forward` | [deepseek_v2.py:2337](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2337) | V2 模型 forward（含 CP split/gather） |
| `DeepseekV2AttentionMLA` | [deepseek_v2.py:1425](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1425) | V2 MLA 注意力层 |
| `rebuild_cp_kv_cache` | [deepseek_v2.py:1874](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1874) | V2 KV latent cache all-gather |
| `DeepseekV2DecoderLayer.forward` | [deepseek_v2.py:2054](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2054) | V2 decoder layer（通过 communicator 通信） |
| `GlmMoeDsaForCausalLM` | [glm4_moe.py:1481](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/glm4_moe.py#L1481) | GLM DSA 入口（继承 `DeepseekV2ForCausalLM`） |
| `is_deepseek_dsa` | [model_config.py:102](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/configs/model_config.py#L102) | 判断模型是否为 DSA 架构 |

---

#### 附录 B：in-seq-split vs round-robin-split 总对比

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
| **AllGather rerange** | `cp_reverse_index` 复杂 rerange（split + 段重排） | `view(cp_size,-1).transpose(0,1).reshape` 简单 transpose |
| **每层同步通信** | 1 次（KV AllGather）+ 2×A2A | 3 次（TP MoE：KV AG + MLP AG + MLP RS）或 1 次+2×A2A（DeepEP） |
| **input_ids 处理** | `input_ids[cp_rank::cp_size]`（deepep 分支） | `reshape(-1,cp_size).T.flatten()`（TP）或 stride（deepep） |
| **piecewise CUDA graph** | 禁用（`disable_piecewise_cuda_graph=True`，indexer assert） | 禁用（同样禁用） |

---

#### 附录 C：DeepSeek V4 路径 vs V2 路径（GLM DSA）对比

| 维度 | V4（`deepseek_v4.py`） | V2（`deepseek_v2.py`，GLM DSA 等） |
|------|------------------------|--------------------------------------|
| 入口类 | `DeepseekV4ForCausalLM` | `DeepseekV2ForCausalLM`（`GlmMoeDsaForCausalLM` 继承） |
| hidden_states 维度 | 3D `[seq, hc_mult, hidden_dim]`（mHC） | 2D `[seq, hidden_dim]` |
| Embedding 后处理 | `unsqueeze(1).repeat(1, hc_mult, 1)` | 无 |
| CP 元数据准备 | 含 `apply_cp_reindex()` + `init_flashmla_related()`（仅 round-robin-split） | 仅 `prepare_context_parallel_metadata()` |
| `input_ids` 处理 | `cp_round_robin_input_ids()` + `input_ids_global` | 不处理 `input_ids`，不传 `input_ids_global` |
| Attention TP 强制 | `__init__` 显式 `tp_rank=0, tp_size=1`（[deepseek_v4.py:258-259](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L258-L259)） | 通过 `get_attention_tp_size()` 隐式为 1（[deepseek_v2.py:1467](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1467)） |
| KV all-gather | `_compute_kv_bf16()` → `cp_all_gather_rerange_output(kv)` | `rebuild_cp_kv_cache()` → `cp_all_gather_rerange_output(latent_cache)` |
| MLP 通信方式 | 显式 `dsa_cp_gather_hidden_states()` / `dsa_cp_reduce_scatter_hidden_states()`（TP）或 A2A assert（deepep） | 通过 `DSACPLayerCommunicator.prepare_mlp()` / `.postprocess_layer()` |
| MLP 接收 input_ids | 是（`input_ids=input_ids, input_ids_global=input_ids_global`） | 否 |
| Residual 管理 | `prev_residual` / `prev_post` / `prev_comb`（mHC） | 独立 `residual` tensor（标准 residual connection） |
| 最后阶段顺序 | all-gather → `hc_head` → `norm` | `norm` → all-gather |
| `logits_processor` 调用 | 含 `hidden_states_before_norm=pre_hc_head` | 不含 `pre_hc_head` |
| 适用模型 | DeepSeek V4 | DeepSeek V3.2, GLM-4.5/4.6/4.7 DSA, Mistral-Large-3 等 |

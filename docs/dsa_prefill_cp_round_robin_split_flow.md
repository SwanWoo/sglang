> SGLang DSA Prefill Context Parallel (Round-Robin-Split) 完整流程
> **配置参数**  
> `--enable-dsa-prefill-context-parallel --attn-cp-size 8 --dsa-prefill-cp-mode round-robin-split`  
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

---

#### Step 0: 参数初始化与校验

服务启动时，`ServerArgs.__post_init__()`（[server_args.py:887](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L887)）经过一系列处理后，在 [server_args.py:1000](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1000) 调用 `_handle_context_parallelism()` 方法（定义于 [server_args.py:3218](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3218)），校验：

- `attn_cp_size = 8`，要求 `tp_size % attn_cp_size == 0`（[server_args.py:3234](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3234)）
- `enable_dsa_prefill_context_parallel = True`（字段声明 [server_args.py:803](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L803)）
- `dsa_prefill_cp_mode = "round-robin-split"`（字段声明 [server_args.py:804](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L804)，合法值 `["in-seq-split", "round-robin-split"]`，定义于 `DSA_PREFILL_CP_SPLIT_CHOICES`，[server_args.py:275](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L275)）
- 两个 flag `enable_prefill_context_parallel` 和 `enable_dsa_prefill_context_parallel` 互斥（[server_args.py:3219-3230](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3219-L3230)）

字段声明（[server_args.py:803-804](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L803-L804)）：

```python
enable_dsa_prefill_context_parallel: bool = False            # 是否启用 DSA prefill CP，默认关闭
dsa_prefill_cp_mode: str = "round-robin-split"               # CP 分割模式，默认 round-robin-split
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

`is_deepseek_dsa()`（[model_config.py:102](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/configs/model_config.py#L102)）对其返回 `True`（需同时满足 config 中存在 `index_topk` 属性）：

```python
def is_deepseek_dsa(config) -> bool:                         # 判断模型是否为 DSA 架构
    return (
        _hf_arch(config)                                     # 从 HF config 取 architecture 名
        in (
            "DeepseekV3ForCausalLM",                         # 所有支持 DSA 的模型架构
            "DeepseekV32ForCausalLM",
            "DeepseekV3ForCausalLMNextN",                    # NextN = 投机解码的小模型
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
            "GlmMoeDsaForCausalLM",                          # GLM DSA 在此列表中
        )
        and _hf_attr(config, "index_topk") is not None       # 还需 config 中存在 index_topk 字段
    )
```

从而启用 DSA attention backend 和 CP 相关配置。参数校验逻辑完全相同，无差异。

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

#### Step 3: Scheduler 组 Batch（关键）

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

1. `PrefillAdder` 从 `waiting_queue` 中取出请求
2. 调用 `req.init_next_round_input(tree_cache)` 做 radix cache 前缀匹配
3. 调用 `adder.add_one_req()` 判断能否加入 batch
4. 得到 `can_run_list`（本 batch 要 prefill 的请求列表）
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

#### Step 4: CP 元数据准备（CausalLM.forward 入口）

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
        if is_dsa_prefill_cp_round_robin_split():            # 如果是 round-robin-split 模式
            attn_backend = get_attn_backend()                # 获取当前 attention backend（如 DSA backend）
            metadata = attn_backend.forward_metadata         # 取出前向元数据
            core_meta = metadata.core_attn_metadata          # 取出核心 attention 元数据
            core_meta.apply_cp_reindex()                     # 对 attention index 做 CP 重排（V4 特有）
            core_meta.init_flashmla_related(is_prefill=True) # 初始化 FlashMLA 相关参数（V4 特有）
            # ...                                             # 还会重建 indexer_metadata
```

其中 `can_dsa_cp_split()`（[dsa/utils.py:175](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L175)）对 round-robin-split 模式检查：

```python
cur_cp_seq_len = seq_len // cp_size                          # 计算每个 rank 分到的序列长度
assert seq_len % cp_size == 0                                # 断言序列长度必须被 cp_size 整除
# 还要求 cur_cp_seq_len != 0, cp_size > 1, use_dsa=True,    # 其他前置条件（略）
# forward_mode.is_context_parallel_extend(), is_dsa_enable_prefill_cp(),
# sum(extend_seq_lens_cpu) >= cp_size
```

`can_dsa_prefill_cp_round_robin_split()`（[dsa/utils.py:85](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L85)）也做类似检查：

```python
def can_dsa_prefill_cp_round_robin_split(forward_batch):     # 判断 forward_batch 能否使用 RR CP
    if not forward_batch.forward_mode.is_context_parallel_extend():  # 非 CP extend 模式直接返回
        return False
    cp_size = get_attention_cp_size()                        # 获取 CP 并行度
    seq_len = sum(forward_batch.extend_seq_lens_cpu)         # 计算 batch 中所有请求的 extend 总长度
    return (
        is_dsa_prefill_cp_round_robin_split()                # 必须是 round-robin-split 模式
        and seq_len > 0                                      # 序列不能为空
        and seq_len >= cp_size                               # 序列长度 ≥ cp_size 才能分割
        and cp_size > 1                                      # cp_size=1 无意义
    )
```

其中 `ForwardMode.is_context_parallel_extend()`（[forward_batch_info.py:125](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/model_executor/forward_batch_info.py#L125)）对 `EXTEND` 和 `MIXED` 模式返回 `True`（也可选择包含 `DRAFT_EXTEND_V2`）。

> **重要**：对于 `"hello"` 这种只有 1 个 token 的短 prompt，`seq_len=1 < cp_size=8`，`can_dsa_cp_split()` 中 `cur_cp_seq_len = 1 // 8 = 0`，CP 不会真正生效。

我们假设 prompt 足够长（比如 64 个 token），这样才能展示完整的 CP 流程。

`prepare_context_parallel_metadata()`（[cp_utils.py:491](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L491)）在 round-robin-split 模式下直接返回空的 `ContextParallelMetadata()`（[cp_utils.py:503-504](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L503-L504)）：

```python
if is_dsa_prefill_cp_round_robin_split():                    # round-robin-split 模式
    return ContextParallelMetadata()                         # 返回空元数据（不需要 zigzag 索引重排）
```

因为 round-robin-split 不需要 zigzag 的复杂索引重排。

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

1. **没有 `apply_cp_reindex()` / `init_flashmla_related()` 调用**：V4 在 `is_dsa_prefill_cp_round_robin_split()` 分支中额外执行 `core_meta.apply_cp_reindex()` 和 `core_meta.init_flashmla_related(is_prefill=True)`（[deepseek_v4.py:1740-1748](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1740-L1748)），V2 路径不做这些操作。
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

#### Step 5: Round-Robin-Split 分发 Token（核心计算）

假设 prompt tokenize 后有 64 个 token：`[t0, t1, t2, ..., t63]`

> **注意**：实际 CP split 操作发生在模型 forward 入口（V4: `DeepseekV4Model.forward`，[deepseek_v4.py:1544](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1544)；V2: `DeepseekV2Model.forward`，[deepseek_v2.py:2337](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2337)），而非 `ForwardBatch.init_new()` 中。所有 rank 先对完整 `input_ids` 做 embedding，然后才执行 split。

##### 5a. Input IDs 的 Round-Robin 分割（仅 V4）

`cp_round_robin_input_ids()`（[cp_utils.py:191-212](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L191-L212)）有两个分支：

```python
def cp_round_robin_input_ids(input_ids):                     # 对 input_ids 做 round-robin 分割
    cp_size = get_attention_cp_size()                        # 获取 CP 并行度
    cp_rank = get_attention_cp_rank()                        # 获取当前 CP rank
    if get_moe_a2a_backend().is_none():                      # 非 a2a 模式
        input_ids = input_ids.reshape(-1, cp_size).T.flatten()  # reshape→转置→展平，所有 rank 得到相同的重排序序列
    else:                                                    # a2a 模式（如 DeepEP）
        input_ids = input_ids[cp_rank::cp_size].contiguous() # 每个 rank 取 stride 切片
    return input_ids
```

- **非 a2a 模式**（`get_moe_a2a_backend().is_none()` 为 `True`）：  
  `reshape(-1, cp_size).T.flatten()` 将序列重排为  
  `[t0, t8, ..., t56, t1, t9, ..., t57, ..., t7, t15, ..., t63]`  
  所有 rank 得到相同的完整 64 token 序列（只是顺序重排），用于后续 MoE routing（`input_ids_global`）。

  > ⚠️ 注意：此模式下 `input_ids` 保持完整 64 个用于 MoE routing，而 `hidden_states` 经 split 后每 rank 只有 8 个——两者数量不对齐，`input_ids` 仅作 routing 索引，不与 `hidden_states` 逐元素对应。  
  `hidden_states` 的 CP split 由 `dsa_cp_round_robin_split_data()`（[dsa/utils.py:98](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L98)）独立完成。

- **a2a 模式**（通常与 `--enable-deep-ep` 一起使用）：  
  `input_ids[cp_rank::cp_size]` 进行 per-rank stride 切片，结果如下：

  | CP rank | 拿到的 token IDs |
  | ------- | ---------------- |
  | 0       | `[t0, t8, t16, t24, t32, t40, t48, t56]` |
  | 1       | `[t1, t9, t17, t25, t33, t41, t49, t57]` |
  | 2       | `[t2, t10, t18, t26, t34, t42, t50, t58]` |
  | ...     | ... |
  | 7       | `[t7, t15, t23, t31, t39, t47, t55, t63]` |

  每个 GPU 只拿到 1/8 的 token，且是按 stride 交错取的（非连续切片）。

  另外，CP 模式下会设置 `input_ids_global = input_ids`（[deepseek_v4.py:1580](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1580)），该变量传递给每层的 MLP 用于 MoE routing。

**▶ GLM DSA / V2 路径差异**：V2 路径的 `DeepseekV2Model.forward()`（[deepseek_v2.py:2378-2383](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2378-L2383)）**没有 `cp_round_robin_input_ids()` 调用**，也没有 `input_ids_global` 变量。V2 的 `DeepseekV2DecoderLayer.forward()`（[deepseek_v2.py:2054](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2054)）签名中不接收 `input_ids` 或 `input_ids_global` 参数，MLP/MoE 调用（[deepseek_v2.py:2118-2124](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2118-L2124)）也不传递 `input_ids_global`。MoE 的 expert routing 依赖 hidden_states 本身而非 input_ids。

##### 5b. Hidden States 和 Position IDs 的分割

**V4 路径**：在 `DeepseekV4Model.forward()` 中（[deepseek_v4.py:1575-1580](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1575-L1580)）：

```python
if dsa_use_prefill_cp(forward_batch):                        # 如果当前 forward 使用 CP
    if self.pp_group.is_first_rank:                          # 仅 pipeline 并行第一个 rank 做 split
        hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)  # 对 hidden_states 做 round-robin stride 切片
    positions = cp_split_and_rebuild_position(forward_batch, positions)  # 对 position_ids 做同样的 stride 切片
    input_ids = cp_round_robin_input_ids(input_ids)          # 对 input_ids 做 round-robin 分割（两分支）
    input_ids_global = input_ids                             # 保存分割后的 input_ids 用于 MoE routing
```

`cp_split_and_rebuild_data()`（[cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145)）在 round-robin-split 模式下的核心路径：

```python
def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):  # hidden_states 的 CP 分割入口
    if is_dsa_prefill_cp_round_robin_split():                # 如果是 round-robin-split 模式
        cp_size = get_attention_cp_size()                    # 获取 CP 并行度
        assert input_.shape[0] % cp_size == 0, ...           # 序列长度必须能被 cp_size 整除
        return dsa_cp_round_robin_split_data(input_)         # 调用 round-robin stride 切片函数
    # 非 round-robin 模式走 zigzag 分割路径（略）             # 其他 CP 模式的分割逻辑
```

`cp_split_and_rebuild_position()`（[cp_utils.py:167](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L167)）结构类似，同样在 round-robin-split 模式下调用 `dsa_cp_round_robin_split_data()`（[dsa/utils.py:98](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L98)）：

```python
def dsa_cp_round_robin_split_data(input_):                   # 对 tensor 做 round-robin stride 切片
    cp_size = get_attention_cp_size()                        # 获取 CP 并行度（如 8）
    cp_rank = get_attention_cp_rank()                        # 获取当前 rank（如 0~7）
    # ...                                                    # 省略 padding 等边界处理
    # 对于能被 cp_size 整除的 tensor：
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()  # reshape→取第 cp_rank 列→连续化
```

对 64 个 token 的 3D tensor `[64, hc_mult, hidden_dim]`（V4）或 2D tensor `[64, hidden_dim]`（V2）：

- V4: `view(-1, 8, hc_mult, hidden_dim)` → `[8, 8, hc_mult, hidden_dim]` → `[:, cp_rank]` → `[8, hc_mult, hidden_dim]`
- V2: `view(-1, 8, hidden_dim)` → `[8, 8, hidden_dim]` → `[:, cp_rank]` → `[8, hidden_dim]`

每个 rank 得到 stride 切片，如 rank 0 得到 `[t0, t8, t16, t24, t32, t40, t48, t56]`。

矩阵变换过程（以 V2 2D tensor `[64, D]` 为例，V4 多一个 hc_mult 维度，原理相同）：

```
原始 hidden_states [64, D]:
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬─────┬─────┬─────┐
│ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │ t6 │ t7 │ t8 │ t9 │ ... │ t62 │ t63 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴─────┴─────┴─────┘

Step 1: view(-1, cp_size, D) → reshape 为 [8, 8, D] 矩阵

          cp_rank →  0     1     2     3     4     5     6     7
        ┌────────────────────────────────────────────────────────┐
 组 0   │  t0  │  t1  │  t2  │  t3  │  t4  │  t5  │  t6  │  t7  │
 组 1   │  t8  │  t9  │ t10  │ t11  │ t12  │ t13  │ t14  │ t15  │
 组 2   │ t16  │ t17  │ t18  │ t19  │ t20  │ t21  │ t22  │ t23  │
 组 3   │ t24  │ t25  │ t26  │ t27  │ t28  │ t29  │ t30  │ t31  │
 组 4   │ t32  │ t33  │ t34  │ t35  │ t36  │ t37  │ t38  │ t39  │
 组 5   │ t40  │ t41  │ t42  │ t43  │ t44  │ t45  │ t46  │ t47  │
 组 6   │ t48  │ t49  │ t50  │ t51  │ t52  │ t53  │ t54  │ t55  │
 组 7   │ t56  │ t57  │ t58  │ t59  │ t60  │ t61  │ t62  │ t63  │
        └────────────────────────────────────────────────────────┘

Step 2: [:, cp_rank] → 每个 rank 取对应列

 rank 0 取第 0 列       rank 1 取第 1 列             rank 7 取第 7 列
 ┌──────────────┐       ┌──────────────┐             ┌──────────────┐
 │  t0  [8, D]  │       │  t1  [8, D]  │             │  t7  [8, D]  │
 │  t8          │       │  t9          │     ...     │ t15          │
 │ t16          │       │ t17          │             │ t23          │
 │ t24          │       │ t25          │             │ t31          │
 │ t32          │       │ t33          │             │ t39          │
 │ t40          │       │ t41          │             │ t47          │
 │ t48          │       │ t49          │             │ t55          │
 │ t56          │       │ t57          │             │ t63          │
 └──────────────┘       └──────────────┘             └──────────────┘

每个 rank 持有的 token 编号 = cp_rank, cp_rank+8, cp_rank+16, ...（stride = cp_size）
```

**▶ GLM DSA / V2 路径差异**：V2 的 `DeepseekV2Model.forward()`（[deepseek_v2.py:2378-2383](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2378-L2383)）CP split 代码：

```python
if dsa_use_prefill_cp(                                       # 检查 DSA CP 是否生效
    forward_batch, self.dsa_enable_prefill_cp
) or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp):  # 或 MLA CP 是否生效
    if self.pp_group.is_first_rank:                          # 仅 PP 第一个 rank 做 split
        hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)  # 对 hidden_states 做 stride 切片
    positions = cp_split_and_rebuild_position(forward_batch, positions)  # 对 positions 做 stride 切片
```

与 V4 对比：只做 `hidden_states` 和 `positions` 的 split，**不调用 `cp_round_robin_input_ids()`**，**不设置 `input_ids_global`**。此外 `dsa_use_prefill_cp` 需要显式传入第二个参数 `self.dsa_enable_prefill_cp`。

##### 5c. extend_seq_lens 的分割

`dsa_cp_round_robin_split_q_seqs()`（[dsa/utils.py:236](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L236)）计算每个 rank 上的 q 序列长度，返回 CPU 和 GPU 两份结果：

```python
def dsa_cp_round_robin_split_q_seqs(                        # 计算每个 rank 上的 q 序列长度
    extend_seqs_cpu, extend_seqs                             # CPU 和 GPU 上的 extend 序列长度
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:         # 返回 CPU/GPU 版本的 q_lens 和 bs_idx
    cp_size = get_attention_cp_size()                        # 获取 CP 并行度
    cp_rank = get_attention_cp_rank()                        # 获取当前 rank
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)  # CPU 路径计算
    ret_q_lens = torch.empty((len(bs_idx_cpu),), device=extend_seqs.device, ...)  # 预分配 GPU 输出 tensor
    bs_idx = torch.empty((len(bs_idx_cpu),), device=extend_seqs.device, ...)      # 预分配 batch index tensor
    dsa_cp_round_robin_split_q_seqs_kernel[grid](           # 调用 Triton kernel 在 GPU 上并行计算
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank  # 传入输入输出和参数
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx   # 返回 CPU 和 GPU 两份结果
```

其中 CPU 路径 `dsa_cp_round_robin_split_q_seqs_cpu()`（[dsa/utils.py:221](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L221)）为核心算法：

```python
def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):       # CPU 路径：计算每 rank 的 q 序列长度
    cp_size = get_attention_cp_size()                        # 获取 CP 并行度（如 8）
    cp_rank = get_attention_cp_rank()                        # 获取当前 rank（如 0~7）
    extra_seq = 0                                            # 累积的余量 token
    q_seqs = []                                              # 每个请求在本 rank 上的 q 长度
    for bs, cur_len in enumerate(extend_seqs):               # 遍历 batch 中每个请求的 extend 长度
        cur_len += extra_seq                                 # 加上前一个请求留下的余量
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)  # 整除 + 余量分配（前 remainder 个 rank 多分 1 个）
        q_seqs.append(cur_seq)                               # 记录本请求在本 rank 上的 q 长度
        extra_seq = cur_len - cur_seq * cp_size              # 计算剩余未分配的 token 数
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])  # 过滤掉分到 0 个 token 的请求
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]       # 只保留有 token 的请求的 q 长度
    return q_seqs, bs_idx                                    # 返回有效 q 长度列表和对应的 batch 索引
```

对于单个 64 token 序列，`cp_size=8`：

- `cur_len = 64`，`cur_seq = 64 // 8 + int(0 > cp_rank) = 8`
- 所有 rank 各处理 8 个 q token

---

#### Step 6: Model Forward（每一层的计算过程）

`DeepseekV4ForCausalLM.forward()`（[deepseek_v4.py:1722](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1722)）调用 `self.model.forward()`（[deepseek_v4.py:1751](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1751)），进入 `DeepseekV4Model.forward()`（[deepseek_v4.py:1544](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1544)）。

**单层 Transformer Block 数据流总览**（每个 rank 视角，cp_size=8, 64 tokens）：

```
输入: hidden_states [8, D]（split 状态，每 rank 持有 1/8 stride tokens）

  ┌──────────────────────────────────────────────────────────────────┐
  │                    Self-Attention                                │
  │                                                                  │
  │  Q: 本 rank 的 8 tokens                  shape = [8, heads, d]  │
  │  K/V: AllGather+Rerange 后的完整 64 tokens   shape = [64, ...]  │
  │                                                                  │
  │  ──→ 每个 rank 独立计算 8 个 Q 对 64 个 KV 的 attention          │
  │  ──→ 输出仍是 split 状态                  shape = [8, D]        │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │                  MLP / MoE（非 a2a 模式）                       │
  │                                                                  │
  │  ① AllGather（无 rerange）   [8, D] × 8 ranks → [64, D]        │
  │  ② MLP/MoE per-token 计算   [64, D] → [64, D]（EP partial）    │
  │  ③ ReduceScatter             [64, D] → [8, D]（sum + split）    │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓
输出: hidden_states [8, D]（split 状态，传入下一层）

通信次数: KV AllGather ×1 + MLP AllGather ×1 + MLP ReduceScatter ×1 = 3 次集合通信/层
```

##### 6a. Embedding 层

```python
hidden_states = self.embed_tokens(input_ids)               # 将 token IDs 映射为 embedding 向量 [64, hidden_dim]
hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)  # mHC 扩展：[64, D] → [64, 1, D] → [64, hc_mult, D]
```

每个 CP rank 上：`hidden_states.shape = [64, hc_mult, hidden_dim]`（所有 rank 相同，因为 embedding 在 split 之前执行）。`unsqueeze` + `repeat` 是 DSV4 特有的 **mHC（multi-Head Composition）扩展**。

接着在 [deepseek_v4.py:1575-1580](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1575-L1580) 执行 CP split（见 Step 5b），分割后每个 rank 上的 `hidden_states.shape` 变为 `[8, hc_mult, hidden_dim]`。

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

即每个 GPU 拥有全部的 attention heads（权重复制），独立计算完整的 attention。

`MQALayer._forward_prepare()`（[deepseek_v4.py:698](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L698)）是 CP 最关键的部分。实际代码有多条分支路径（`fuse_wqa_wkv`、`use_fused_qk_norm_rope` 等），以下为 CP 相关核心逻辑：

```python
# 1. 计算 Q（每个 rank 独立计算自己的 Q）
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

###### ii. `cp_all_gather_rerange_output()` 的计算细节

`cp_all_gather_rerange_output()`（[cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310)）执行流程：

```python
def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):  # AllGather + rerange 恢复顺序
    if is_dsa_prefill_cp_round_robin_split():                # round-robin-split 模式
        with use_symmetric_memory(                           # 在 symmetric memory 上下文中分配 output
            get_attention_cp_group(), disabled=not is_allocation_symmetric()  # 若启用 symmetric memory 则使用对称分配
        ):
            output_tensor = input_tensor.new_empty(          # 预分配 cp_size 倍大小的输出 tensor
                (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),  # [8*8, ...] = [64, ...]
            )
        # 1. All-gather: 每个 rank 的 [8, ...] → [64, ...]
        attn_cp_all_gather_into_tensor(output_tensor, input_tensor)  # NCCL AllGather 集合通信
        # 此时 output_tensor = [rank0_kv, rank1_kv, ..., rank7_kv]（交错排列）

        # 2. Rerange: 交错重排回原始顺序
        out_shape = output_tensor.shape                      # 保存原始 shape
        output_tensor = (
            output_tensor.view(cp_size, -1, *out_shape[1:])   # [8, 8, ...] — 按 rank 分组
            .transpose(0, 1)                                  # [8, 8, ...] — 行列互换，rank↔token_group
            .reshape(out_shape)                               # [64, ...] — 展平回一维，顺序恢复
        )
        # 现在: [t0_kv, t1_kv, t2_kv, ..., t63_kv] ← 正确顺序！
        return output_tensor                                 # 返回顺序正确的完整 tensor
```

通信开销：NCCL AllGather，每个 rank 发送 `8 * hc_mult * kv_dim` 数据，总共接收 `64 * hc_mult * kv_dim`。

矩阵变换过程（以 cp_size=8, 64 tokens 为例）：

```
Step 1: AllGather — 每个 rank 的 KV 拼接到一起

  rank 0       rank 1       rank 2            rank 7
  ┌────────┐  ┌────────┐  ┌────────┐        ┌────────┐
  │t0  _kv │  │t1  _kv │  │t2  _kv │        │t7  _kv │
  │t8  _kv │  │t9  _kv │  │t10 _kv │  ...   │t15 _kv │
  │t16 _kv │  │t17 _kv │  │t18 _kv │        │t23 _kv │
  │  ...   │  │  ...   │  │  ...   │        │  ...   │
  │t56 _kv │  │t57 _kv │  │t58 _kv │        │t63 _kv │
  └────────┘  └────────┘  └────────┘        └────────┘
       ↓ AllGather → 直接拼接 [rank0 | rank1 | ... | rank7]
  ┌──────────────────────────────────────────────────────────┐
  │ t0,t8,..,t56, t1,t9,..,t57, t2,t10,..,t58, ..., t63     │ shape=[64, kv_dim]
  └──────────────────────────────────────────────────────────┘

Step 2: view(cp_size, -1, ...) → [8, 8, kv_dim]

            token_idx →  0     1     2     3     4     5     6     7
  rank 0 块 (row 0)   │ t0  │ t8  │ t16 │ t24 │ t32 │ t40 │ t48 │ t56 │
  rank 1 块 (row 1)   │ t1  │ t9  │ t17 │ t25 │ t33 │ t41 │ t49 │ t57 │
  rank 2 块 (row 2)   │ t2  │ t10 │ t18 │ t26 │ t34 │ t42 │ t50 │ t58 │
  rank 3 块 (row 3)   │ t3  │ t11 │ t19 │ t27 │ t35 │ t43 │ t51 │ t59 │
  rank 4 块 (row 4)   │ t4  │ t12 │ t20 │ t28 │ t36 │ t44 │ t52 │ t60 │
  rank 5 块 (row 5)   │ t5  │ t13 │ t21 │ t29 │ t37 │ t45 │ t53 │ t61 │
  rank 6 块 (row 6)   │ t6  │ t14 │ t22 │ t30 │ t38 │ t46 │ t54 │ t62 │
  rank 7 块 (row 7)   │ t7  │ t15 │ t23 │ t31 │ t39 │ t47 │ t55 │ t63 │

Step 3: transpose(0, 1) → 行列互换

            rank_idx →   0     1     2     3     4     5     6     7
  token 组 0 (row 0)   │ t0  │ t1  │ t2  │ t3  │ t4  │ t5  │ t6  │ t7  │
  token 组 1 (row 1)   │ t8  │ t9  │ t10 │ t11 │ t12 │ t13 │ t14 │ t15 │
  token 组 2 (row 2)   │ t16 │ t17 │ t18 │ t19 │ t20 │ t21 │ t22 │ t23 │
     ...
  token 组 7 (row 7)   │ t56 │ t57 │ t58 │ t59 │ t60 │ t61 │ t62 │ t63 │

Step 4: reshape(64, ...) → 按行展平

  ┌──────────────────────────────────────────────────────────┐
  │ t0, t1, t2, t3, ..., t62, t63                            │ shape=[64, kv_dim]
  └──────────────────────────────────────────────────────────┘
  ✓ 恢复为原始 token 顺序！
```

###### iii. DSA Indexer（如果有 C4Indexer）

```python
if self.indexer is not None:                                 # 如果配置了 DSA Indexer（C4Indexer 等）
    self.indexer(x, q_lora, forward_batch, attn_backend)     # 执行稀疏 attention 的索引计算
```

DSA Indexer 内部也会调用 `cp_all_gather_rerange_output()`（调用点 [dsa_indexer.py:509](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L509)、`:523`、`:1721`），确保看到完整序列。

###### iv. Attention Forward（DSA Backend）

```python
attn_output = attn_backend.forward(q, ...)                   # Q=[8,heads,d] 对 KV=[64,...] 做 attention 计算
```

> **关键**：Q 只有 1/8，但 KV 是完整的！每个 rank 计算一部分 Q 的 attention，但需要看到所有 KV。由于 TP=1，每个 GPU 独立完成 attention，无需 TP 通信。

Attention 输出后，执行 DSV4 特有的 **inverse RoPE** 操作。

###### v. Attention 输出保持 split 状态

Attention output 保持 split 状态（每个 rank 只有 8 个 token 的结果），不立即 all-gather。

###### vi. MLP / MoE 层（带通信）

MLP 层的处理在 `DeepseekV4DecoderLayer.forward()` 中（[deepseek_v4.py:1387-1428](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1387-L1428)）：

```python
_use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)  # 判断本层是否使用 CP

if _use_cp:                                                  # CP 启用时
    if get_moe_a2a_backend().is_none():                      # 非 a2a 模式
        hidden_states = dsa_cp_gather_hidden_states(hidden_states)  # AllGather: [8,D] → [64,D]（无 rerange）
    else:                                                    # a2a 模式（DeepEP）
        assert get_moe_a2a_backend().is_deepep()             # 断言必须是 DeepEP 后端

hidden_states = self.mlp(hidden_states, ..., use_reduce_scatter=_use_cp)  # MLP/MoE 前向计算，CP 时跳过内部 all-reduce

if _use_cp and get_moe_a2a_backend().is_none():              # CP + 非 a2a 模式
    hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)  # ReduceScatter: [64,D] → [8,D]（sum + split）
```

具体细节：

- **MLP 前 AllGather**：`dsa_cp_gather_hidden_states()`（[communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55)）

  ```python
  def dsa_cp_gather_hidden_states(hidden_states):            # MLP 前 AllGather（无 rerange）
      attn_dp_size = get_attention_dp_size()                 # 获取 attention DP size
      attn_tp_size = get_attention_tp_size()                 # 获取 attention TP size
      assert attn_dp_size == 1 and attn_tp_size == 1         # 非 a2a 路径要求 DP=1 且 TP=1
      hidden_states, local_hidden_states = (                 # 交换：hidden_states 指向预分配 buffer
          get_local_dp_buffer(get_attention_cp_group()),      # 获取 CP group 的预分配接收 buffer
          hidden_states,                                     # local_hidden_states 保存本 rank 原始数据
      )
      attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)  # NCCL AllGather 集合通信
      return hidden_states                                   # 返回包含所有 rank 数据的完整 buffer [64,D]
  ```

  注意：此处为纯 AllGather，无 rerange。AllGather 后 buffer 内容为 `[rank0_data, rank1_data, ..., rank7_data]`，即 `[t0, t8, ..., t56, t1, t9, ..., t57, ..., t7, t15, ..., t63]`（round-robin 交错排列）。由于 MLP 是 per-token 操作，token 顺序不影响计算结果。

  注意：此函数还有 `assert attn_dp_size == 1 and attn_tp_size == 1`，即非 a2a 路径下 CP 要求 `dp_size=1` 且 `attn_tp_size=1`。

  MLP 通信全流程（非 a2a 模式）：

  ```
  ┌─── Attn 输出（split 状态）──────────────────────────────────────────┐
  │ rank 0: [t0, t8, t16, t24, t32, t40, t48, t56]   shape=[8, D]     │
  │ rank 1: [t1, t9, t17, t25, t33, t41, t49, t57]                    │
  │  ...                                                                │
  │ rank 7: [t7, t15, t23, t31, t39, t47, t55, t63]                   │
  └────────────────────────────────────────────────────────────────────┘
                               ↓ AllGather（无 rerange）
  ┌─── 每个 rank 都拿到完整 64 tokens（交错排列）─────────────────────┐
  │ [t0,t8,..,t56 | t1,t9,..,t57 | ... | t7,t15,..,t63] = [64, D]    │
  │  ↑ rank0 块     ↑ rank1 块                ↑ rank7 块              │
  │  顺序 ≠ 原始，但 MLP 是 per-token 操作，不影响结果                │
  └────────────────────────────────────────────────────────────────────┘
                               ↓ MLP / MoE（per-token，TP partial result，默认 ep_size=1）
  ┌─── MLP 输出（每 rank 持有 TP partial result）─────────────────────┐
  │ [r0,r8,..,r56 | r1,r9,..,r57 | ... | r7,r15,..,r63] = [64, D]    │
  └────────────────────────────────────────────────────────────────────┘
                               ↓ ReduceScatter = reduce + split
  ┌─── 回到 split 状态（每 rank 得到自己 stride 的完整 MLP 结果）────┐
  │ rank 0: [r0, r8, r16, r24, r32, r40, r48, r56]   shape=[8, D]    │
  │ rank 1: [r1, r9, r17, r25, r33, r41, r49, r57]   （r = sum of   │
  │  ...                                                partial result)│
  │ rank 7: [r7, r15, r23, r31, r39, r47, r55, r63]                  │
  └────────────────────────────────────────────────────────────────────┘
  ```

- **MLP 计算**：  
  - 默认 `ep_size=1`，所有 expert 在所有 rank 上均存在，但每个 expert 的权重按 TP 切分（`moe_tp_size=tp_size`），输出为 TP partial result。  
    `use_reduce_scatter=True` 使 `should_skip_post_experts_all_reduce()`（[moe/utils.py:422](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L422)）返回 `True`（line 446-447），跳过 MoE 内部的 post-experts TP all-reduce（由后续 CP ReduceScatter 替代）：

    ```python
    def should_skip_post_experts_all_reduce(           # 判断是否跳过 MoE 内部的 post-experts all-reduce
        *, is_tp_path: bool,                           # 是否走 TP 路径
        use_reduce_scatter: bool = False,              # 是否使用 ReduceScatter（CP 时为 True）
        should_allreduce_fusion: bool = False,         # 是否使用 allreduce fusion
    ) -> bool:
        if should_allreduce_fusion or use_reduce_scatter:  # CP 时 use_reduce_scatter=True → 跳过内部 all-reduce
            return True                                    # 因为后续会由 dsa_cp_reduce_scatter 统一处理
        if should_use_dp_reduce_scatterv():            # DP reduce-scatter 模式也跳过
            return True
        if is_tp_path and should_use_flashinfer_cutlass_moe_fp4_allgather():  # flashinfer MoE FP4 特殊处理
            return True
        return False                                   # 默认不跳过（正常做 all-reduce）
    ```

- **MLP 后 ReduceScatter**：`dsa_cp_reduce_scatter_hidden_states()`（[communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67)）

  ```python
  def dsa_cp_reduce_scatter_hidden_states(hidden_states):    # MLP 后 ReduceScatter
      attn_dp_size = get_attention_dp_size()                 # 获取 attention DP size
      attn_tp_size = get_attention_tp_size()                 # 获取 attention TP size
      assert attn_dp_size == 1 and attn_tp_size == 1         # 断言 DP=1 且 TP=1
      cp_size = get_attention_cp_size()                      # 获取 CP 并行度（如 8）
      cp_rank = get_attention_cp_rank()                      # 获取当前 rank（如 0~7）
      input_hidden_states = hidden_states                    # 保存完整输入 [64,D] 的引用
      hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]  # 取第 cp_rank 块作为输出目标 [8,D]
      attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)  # NCCL ReduceScatter：sum + 分发到各 rank
      return hidden_states                                   # 返回本 rank 的完整 MLP 结果 [8,D]
  ```

  - AllGather 后的 buffer 布局是 round-robin 交错的：`[t0, t8, ..., t56, t1, t9, ..., t57, ...]`  
  - `tensor_split(8)[cp_rank]` 取第 `cp_rank` 块，即 rank 0 取前 1/8 元素 `[t0, t8, t16, t24, t32, t40, t48, t56]`——这是 rank 0 原本持有的 stride token。  
  - ReduceScatter 对各 rank 的 TP partial result 求和（替代 MoE 内部的 TP AllReduce）并按 rank 分割，每个 rank 最终得到自己负责的那组 stride token 的完整 MLP 输出。

- **a2a 模式（DeepEP）**：不走上述 AllGather/ReduceScatter 路径。  
  a2a 模式下各 rank 的 `input_ids` 已经是 per-rank stride 切片（见 Step 5a `cp_round_robin_input_ids` 的 a2a 分支），MoE 通信由 DeepEP 的 all-to-all 机制处理。此时 `DeepseekV4DecoderLayer` 走 line 1402-1406 的断言分支：

  ```python
  assert get_moe_a2a_backend().is_deepep(), (               # a2a 模式下必须使用 DeepEP 后端
      "CP requires DeepEP (moe_a2a_backend == deepep). "     # 错误信息：CP 需要 DeepEP
      "Only DeepEP is tested with CP's per-rank token split."  # 只有 DeepEP 经过测试
  )
  ```

- **为什么 round-robin-split 默认走 AllGather 而非跳过**：

  **默认配置下 EP 并未开启**。round-robin-split 默认 `ep_size=1`（[server_args.py:635](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L635)），`moe_a2a_backend="none"`（[server_args.py:636](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L636)）。对比 in-seq-split 模式在 [server_args.py:1865-1866](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1865-L1866) 会自动强制设置 `moe_a2a_backend="deepep"` 和 `ep_size=tp_size`，round-robin-split 在 [server_args.py:1870-1875](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1870-L1875) **什么都没有设置**：

  ```python
  # server_args.py:635 — 默认值
  ep_size: int = 1                                             # 默认 ep_size=1，不开 EP

  # server_args.py:1861-1875 — in-seq-split 自动设置 vs round-robin-split 不设置
  if self.dsa_prefill_cp_mode == "in-seq-split":
      self.moe_a2a_backend = "deepep"                          # in-seq-split 自动设 a2a=deepep
      self.ep_size = self.tp_size                              # in-seq-split 自动开 EP
  else:                                                        # round-robin-split 分支
      self.enable_dp_attention = True                          # 只设置了 DP attention
      self.moe_dense_tp_size = 1                               # 和 dense_tp_size=1
      # 注意：没有设置 moe_a2a_backend 和 ep_size！
  ```

  **AllGather 的真正原因不是 EP routing，而是 MoE 的 TP 权重切分**。`ep_size=1` 意味着每个 rank 持有**全部 expert**，但 expert 的权重仍然按 TP 切分在各 rank 上。关键推导链：

  ```python
  # distributed/parallel_state.py:2001-2003 — moe_tp_size 的计算
  moe_ep_size = expert_model_parallel_size                     # = ep_size = 1
  moe_dp_size = moe_data_model_parallel_size                   # = 1（默认）
  moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size  # = tp_size / 1 / 1 = tp_size
  # 结论：ep_size=1 时 moe_tp_size = tp_size，每个 expert 的权重按 TP 切分在所有 rank 上
  ```

  所以每个 rank 虽然持有全部 expert，但每个 expert 只持有 $1/P$ 的权重（$w_1, w_2, w_3$ 在 intermediate_size 维度切分），计算出的是 **TP partial result**。正常（非 CP）MoE 前向结束时会做 TP AllReduce 合并 partial results。TP AllReduce 在 MoE class 的 forward 方法中执行（不在底层 `FusedMoE` 层——底层传了 `reduce_results=False`）：

  ```python
  # deepseek_v2.py:1029-1034 (DeepseekV2MoE.forward_normal) — 正常 TP AllReduce
  if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # tp_size > 1 成立
      is_tp_path=True,
      use_reduce_scatter=use_reduce_scatter,         # 正常情况 False → should_skip 返回 False → 执行 all-reduce
      should_allreduce_fusion=should_allreduce_fusion,
  ):
      final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 合并各 rank 的 TP partial result

  # glm4_moe.py:645-650 (ChatGLMMoE.forward_normal) — GLM 同样逻辑
  if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
      is_tp_path=True,
      use_reduce_scatter=use_reduce_scatter,
      should_allreduce_fusion=should_allreduce_fusion,
  ):
      final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
  ```

  但 CP 通过 `use_reduce_scatter=True` 跳过了这个 AllReduce，**改用 AllGather + ReduceScatter 替代**：

  ```python
  # 当 CP 传入 use_reduce_scatter=True 时：
  # should_skip_post_experts_all_reduce(use_reduce_scatter=True) → 返回 True（moe/utils.py:446-447）
  # 整个 if 不成立 → TP AllReduce 被跳过！
  # 后续由 CP 的 ReduceScatter 统一完成 TP reduce + CP scatter
  ```

  **为什么要替代？** 因为 CP 的 AllGather 把 split 状态（$N/P$ tokens）扩展为完整的 $N$ tokens 后再过 MoE，后续的 ReduceScatter **同时完成了 TP reduce（求和 partial results）和 CP scatter（回到 split 状态）**。这将 "TP AllReduce + 维持 split 状态" 融合成一次 ReduceScatter 通信。

  **代码路径分析**：round-robin-split 模式在 [server_args.py:1870-1875](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1870-L1875) 中**没有强制设置 `moe_a2a_backend = "deepep"`**（对比 in-seq-split 模式在 `:1865` 强制设置了）。`moe_a2a_backend` 默认值为 `"none"`（[server_args.py:635](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L635)），`ep_size` 默认值为 `1`（同行）。因此只要用户没有手动传 `--moe-a2a-backend deepep`（及 `--ep-size`），就一定走 AllGather 分支。

  **如何跳过**：启动时加 `--moe-a2a-backend deepep --ep-size <tp_size>`。此时 V4 路径走 [deepseek_v4.py:1402-1406](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1402-L1406) 的 assert 分支（跳过 AllGather），V2 路径的 `_compute_mlp_mode()`（[communicator.py:378](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator.py#L378)，SCATTERED 返回于 `:380-382`）因为 `not get_moe_a2a_backend().is_none()` 为 `True` 返回 `ScatterMode.SCATTERED`（不 gather）。当 `ep_size=tp_size` 时，`moe_tp_size = tp_size / tp_size / 1 = 1`，expert 不再做 TP 切分——每个 rank 只持有 $1/P$ 的 experts 但持有完整权重（EP 模式），token 的跨 rank dispatch 由 DeepEP all-to-all 处理。

  **通信次数对比**：

  | 模式 | MLP AllGather | MLP ReduceScatter | 每层集合通信 |
  |------|:---:|:---:|:---:|
  | 非 a2a（默认） | 1 次 | 1 次 | 3 次（KV AG + MLP AG + MLP RS） |
  | a2a（DeepEP） | 0 次 | 0 次 | 1 次（仅 KV AG，MoE 由 DeepEP a2a 处理） |

- **DSACPLayerCommunicator（GLM / V2 路径）**：GLM（`glm4_moe.py`）和 V2（`deepseek_v2.py`）不直接调用 `dsa_cp_gather_hidden_states` / `dsa_cp_reduce_scatter_hidden_states`，而是通过 `DSACPLayerCommunicator`（[communicator_dsa_cp.py:79](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L79)）间接调用。流程完全等价：

  1. `prepare_mlp()` → 当 `mlp_mode=FULL` 时（[communicator.py:391](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator.py#L391)，默认 non-a2a 路径），调用 `_gather_hidden_states_and_residual()`（[communicator_dsa_cp.py:170](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L170)），内部执行 `dsa_cp_gather_hidden_states()`（`:185`）
  2. `should_use_reduce_scatter()` → DSA CP 时返回 `True`（[communicator.py:718-719](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator.py#L718-L719)），传入 MoE forward 跳过 TP AllReduce
  3. `postprocess_layer()` → 调用 `_scatter_hidden_states()`（[communicator_dsa_cp.py:220](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L220)），内部执行 `dsa_cp_reduce_scatter_hidden_states()`（`:230`）

  GLM 和 V2 的 MoE 层（[glm4_moe.py:645](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/glm4_moe.py#L645)、[deepseek_v2.py:1029](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1029)）都有相同的 `should_skip_post_experts_all_reduce` 判断，CP 时跳过 TP AllReduce。

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
hidden_states, residual = self.layer_communicator.prepare_mlp(      # MLP 前处理：内部执行 AllGather + LayerNorm
    hidden_states, residual, forward_batch                          # 将 split 状态恢复为完整序列供 MLP 使用
)
use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(  # 判断是否使用 ReduceScatter
    forward_batch                                                   # CP 启用时返回 True
)
hidden_states = self.mlp(                                           # MLP/MoE 前向计算
    hidden_states, forward_batch,                                   # 输入完整序列的 hidden_states
    should_allreduce_fusion, use_reduce_scatter,                    # use_reduce_scatter=True 跳过内部 all-reduce
    gemm_output_zero_allocator,                                     # GEMM 输出的零值分配器
)
hidden_states, residual = self.layer_communicator.postprocess_layer(  # MLP 后处理：内部执行 ReduceScatter
    hidden_states, residual, forward_batch                          # 将完整序列重新切分回 split 状态
)
```

关键区别：
1. V2 的 gather/scatter 由 `DSACPLayerCommunicator` 的 `prepare_mlp()` 和 `postprocess_layer()` 内部处理，而非 V4 的显式 `dsa_cp_gather_hidden_states()` / `dsa_cp_reduce_scatter_hidden_states()` 调用。
2. V2 的 MLP 调用（[deepseek_v2.py:2118-2124](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2118-L2124)）**不传递 `input_ids` 或 `input_ids_global`**——V4 的 `self.mlp(hidden_states, ..., input_ids=input_ids, input_ids_global=input_ids_global, use_reduce_scatter=_use_cp)` 在 V2 中简化为 `self.mlp(hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter, ...)`。
3. V2 有独立的 `residual` tensor 贯穿整个 decoder layer（标准 residual connection），V4 使用 `prev_residual` / `prev_post` / `prev_comb`（mHC 特有）。

###### vii. 进入下一层

经过 MLP 后，`hidden_states` 再次变成 split 状态，每个 rank 拥有对应的 stride token 组（如 rank 0: `[t0, t8, t16, ...]`）。下一层重复上述过程。

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

此时 `hidden_states` 恢复为完整序列 `[64, hc_mult, hidden_dim]`（token 顺序正确，经 rerange 恢复）。

```
最终恢复流程（V4 路径）:

  rank 0          rank 1                 rank 7
  [8,hc,D]       [8,hc,D]     ...      [8,hc,D]
  t0,t8,...,t56   t1,t9,...,t57          t7,t15,...,t63
       │               │                     │
       └───────── AllGather ──────────────────┘
                       ↓
              [64, hc_mult, hidden_dim]  (交错: rank0|rank1|...|rank7)
                       ↓ view(8,8,...).transpose(0,1).reshape(64,...)
              [64, hc_mult, hidden_dim]  (正序: t0,t1,t2,...,t63) ✓
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

其中 `self.lm_head` 是 `ParallelLMHead`（初始化于 [deepseek_v4.py:1657](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1657)），`self.logits_processor` 是 `LogitsProcessor`（初始化于 [deepseek_v4.py:1666](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1666)）。`LogitsProcessor` 负责：对 `hidden_states` 投影到 vocab 维度、提取需要采样的 token 位置的 logits、应用 `sampling_params` 等。这一步不是简单的 `logits[-1:]`，而是根据 batch 中各请求的位置提取对应 logits。

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

  Decode 阶段 `ForwardMode.DECODE` 不匹配以上任何条件，因此 `dsa_use_prefill_cp()` 返回 `False`，CP 不生效。

- 每个 rank 独立处理 1 个 token 的 attention（使用完整的 KV cache）
- 直到生成 EOS 或达到 `max_new_tokens`

---

#### 附录 A：关键函数索引

| 函数名 | 文件位置 | 作用 |
|--------|----------|------|
| `_handle_context_parallelism` | [server_args.py:3218](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L3218) (调用于 `:1000`) | 参数校验 |
| `is_dsa_prefill_cp_round_robin_split` | [dsa/utils.py:78](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L78) | 判断是否为 round-robin-split 模式 |
| `can_dsa_prefill_cp_round_robin_split` | [dsa/utils.py:85](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L85) | 判断 forward_batch 能否启用 RR CP |
| `can_dsa_cp_split` | [dsa/utils.py:175](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L175) | 判断序列能否做 CP split |
| `dsa_use_prefill_cp` | [dsa/utils.py:265](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L265) | 判断当前 forward 是否使用 CP |
| `dsa_cp_round_robin_split_data` | [dsa/utils.py:98](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L98) | 对 tensor 做 round-robin stride 切片 |
| `cp_split_and_rebuild_data` | [cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145) | hidden_states 的 CP 分割入口 |
| `cp_split_and_rebuild_position` | [cp_utils.py:167](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L167) | position_ids 的 CP 分割入口 |
| `cp_round_robin_input_ids` | [cp_utils.py:191](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L191) | input_ids 的 CP 分割（两分支） |
| `cp_all_gather_rerange_output` | [cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310) | AllGather + rerange 恢复顺序 |
| `dsa_cp_gather_hidden_states` | [communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55) | MLP 前 AllGather（无 rerange） |
| `dsa_cp_reduce_scatter_hidden_states` | [communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67) | MLP 后 ReduceScatter |
| `prepare_context_parallel_metadata` | [cp_utils.py:491](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L491) | 构建 CP 元数据（RR 返回空） |
| `dsa_cp_round_robin_split_q_seqs_cpu` | [dsa/utils.py:221](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L221) | 计算每 rank 的 q 序列长度 |
| `MQALayer.__init__` | [deepseek_v4.py:243](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L243) | CP 下 TP 强制为 1（`:253-259`） |
| `MQALayer._forward_prepare` | [deepseek_v4.py:698](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L698) | Q/KV 计算 + KV all-gather |
| `DeepseekV4Model.forward` | [deepseek_v4.py:1544](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1544) | 模型 forward（含 CP split/gather） |
| `DeepseekV4ForCausalLM.forward` | [deepseek_v4.py:1722](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1722) | 顶层 forward（CP 元数据准备 + logits） |
| `should_skip_post_experts_all_reduce` | [moe/utils.py:422](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L422) | CP 下跳过 MoE 内部 all-reduce |
| `get_attention_cp_rank` | [dp_attention.py:338](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L338) | 获取当前 CP rank |
| `get_attention_cp_size` | [dp_attention.py:342](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L342) | 获取 CP size |
| `attn_cp_all_gather_into_tensor` | [dp_attention.py:599](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L599) | 对 CP group 执行 AllGather |
| `attn_cp_reduce_scatter_tensor` | [dp_attention.py:587](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/dp_attention.py#L587) | 对 CP group 执行 ReduceScatter |
| `DeepseekV2ForCausalLM.forward` | [deepseek_v2.py:2633](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2633) | V2 顶层 forward（GLM DSA 继承） |
| `DeepseekV2Model.forward` | [deepseek_v2.py:2337](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2337) | V2 模型 forward（含 CP split/gather） |
| `DeepseekV2AttentionMLA` | [deepseek_v2.py:1425](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1425) | V2 MLA 注意力层 |
| `rebuild_cp_kv_cache` | [deepseek_v2.py:1874](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1874) | V2 KV latent cache all-gather |
| `DeepseekV2DecoderLayer.forward` | [deepseek_v2.py:2054](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2054) | V2 decoder layer（通过 communicator 通信） |
| `GlmMoeDsaForCausalLM` | [glm4_moe.py:1481](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/glm4_moe.py#L1481) | GLM DSA 入口（继承 `DeepseekV2ForCausalLM`） |
| `is_deepseek_dsa` | [model_config.py:102](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/configs/model_config.py#L102) | 判断模型是否为 DSA 架构 |

---

#### 附录 B：两种模式的通信对比

| | 非 a2a 模式 (`moe_a2a_backend=none`) | a2a 模式 (DeepEP) |
|--|--------------------------------------|-------------------|
| `input_ids` | 全 rank 相同（重排序后完整 64 token） | 每 rank stride 切片（8 token） |
| `hidden_states` | 每 rank stride 切片（8 token） | 每 rank stride 切片（8 token） |
| KV all-gather | `cp_all_gather_rerange_output` | `cp_all_gather_rerange_output` |
| MLP 前 | `dsa_cp_gather_hidden_states` | 不执行（DeepEP 自行处理） |
| MLP 后 | `dsa_cp_reduce_scatter_hidden_states` | 不执行（DeepEP 自行处理） |
| MoE 内部 AR | 跳过（`use_reduce_scatter=True`） | 跳过（`use_reduce_scatter=True`） |

---

#### 附录 C：DeepSeek V4 路径 vs V2 路径（GLM DSA）对比

| 维度 | V4（`deepseek_v4.py`） | V2（`deepseek_v2.py`，GLM DSA 等） |
|------|------------------------|--------------------------------------|
| 入口类 | `DeepseekV4ForCausalLM` | `DeepseekV2ForCausalLM`（`GlmMoeDsaForCausalLM` 继承） |
| hidden_states 维度 | 3D `[seq, hc_mult, hidden_dim]`（mHC） | 2D `[seq, hidden_dim]` |
| Embedding 后处理 | `unsqueeze(1).repeat(1, hc_mult, 1)` | 无 |
| CP 元数据准备 | 含 `apply_cp_reindex()` + `init_flashmla_related()` | 仅 `prepare_context_parallel_metadata()` |
| `input_ids` 处理 | `cp_round_robin_input_ids()` + `input_ids_global` | 不处理 `input_ids`，不传 `input_ids_global` |
| Attention TP 强制 | `__init__` 显式 `tp_rank=0, tp_size=1`（[deepseek_v4.py:258-259](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L258-L259)） | 通过 `get_attention_tp_size()` 隐式为 1（[deepseek_v2.py:1467](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1467)） |
| KV all-gather | `_compute_kv_bf16()` → `cp_all_gather_rerange_output(kv)` | `rebuild_cp_kv_cache()` → `cp_all_gather_rerange_output(latent_cache)` |
| MLP 通信方式 | 显式 `dsa_cp_gather_hidden_states()` / `dsa_cp_reduce_scatter_hidden_states()` | 通过 `DSACPLayerCommunicator.prepare_mlp()` / `.postprocess_layer()` |
| MLP 接收 input_ids | 是（`input_ids=input_ids, input_ids_global=input_ids_global`） | 否 |
| Residual 管理 | `prev_residual` / `prev_post` / `prev_comb`（mHC） | 独立 `residual` tensor（标准 residual connection） |
| 最后阶段顺序 | all-gather → `hc_head` → `norm` | `norm` → all-gather |
| `logits_processor` 调用 | 含 `hidden_states_before_norm=pre_hc_head` | 不含 `pre_hc_head` |
| 适用模型 | DeepSeek V4 | DeepSeek V3.2, GLM-4.5/4.6/4.7 DSA, Mistral-Large-3 等 |

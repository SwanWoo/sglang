> DSA Prefill Context Parallel (Round-Robin-Split) 数学原理
> 
> 本文档从数学角度严格描述 SGLang 中 DSA Prefill Context Parallel（round-robin-split 模式）的核心算法。
> 所有公式均基于实际代码推导，具体代码流程参见 [`dsa_prefill_cp_round_robin_split_flow.md`](dsa_prefill_cp_round_robin_split_flow.md)。
>
> **适用模型**：DeepSeek V4（`deepseek_v4.py`，mHC 族）及 DeepSeek V2 路径（`deepseek_v2.py`，MLA 族，包括 GLM DSA、V3.2 等）。
> V2 路径与 V4 路径的差异在各节中以 "**▶ V4 mHC 扩展**" 标注。

---

#### 1. 符号定义

| 符号 | 含义 |
|------|------|
| $N$ | 序列总长度（token 数） |
| $P$ | CP 并行度（GPU 数），即 `--attn-cp-size` 的值 |
| $r$ | 当前 GPU 的 CP rank，$r \in \{0, 1, \ldots, P-1\}$ |
| $D$ | hidden dimension |
| $D_{kv}$ | KV 维度（V2 为 $D_{\text{kv\_lora}} + D_{\text{rope}}$；V4 为统一的 kv_dim） |
| $M$ | mHC 乘数（仅 V4，V2 路径 $M=1$） |
| $n_h$ | attention head 数 |
| $d_h$ | 每个 head 的维度 |
| $L$ | Transformer 层数 |
| $\mathbf{H} \in \mathbb{R}^{N \times D}$ | hidden\_states 矩阵 |
| $\mathbf{h}_i \in \mathbb{R}^{D}$ | 第 $i$ 个 token 的 hidden state 向量（$\mathbf{H}$ 的第 $i$ 行） |
| $\mathbf{p} \in \mathbb{Z}^{N}$ | position IDs 向量，$\mathbf{p}[i] = i$ |

**约束**：$N \bmod P = 0$（序列长度必须被 CP 并行度整除）。

---

#### 2. Round-Robin Split 算子

##### 2.1 定义

对应 `dsa_cp_round_robin_split_data()`（[dsa/utils.py:98](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L98)）和 `cp_split_and_rebuild_data()`（[cp_utils.py:145](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L145)）。

给定 $\mathbf{H} \in \mathbb{R}^{N \times D}$，定义 **Round-Robin Split 算子** $\text{Split}_r$：

$$\text{Split}_r(\mathbf{H}) = \mathbf{H}^{(r)} \in \mathbb{R}^{(N/P) \times D}$$

其中 $\mathbf{H}^{(r)}$ 的第 $j$ 行为：

$$\mathbf{H}^{(r)}[j, :] = \mathbf{H}[j \cdot P + r, :], \quad j = 0, 1, \ldots, \frac{N}{P} - 1$$

即 rank $r$ 取原始序列中**下标模 $P$ 等于 $r$** 的所有 token：

$$\text{rank } r \text{ 持有的 token 下标集合} = \{r,\; r+P,\; r+2P,\; \ldots,\; r + (\tfrac{N}{P}-1) \cdot P\}$$

##### 2.2 矩阵形式推导

将 $\mathbf{H} \in \mathbb{R}^{N \times D}$ 重塑为三维张量 $\hat{\mathbf{H}} \in \mathbb{R}^{(N/P) \times P \times D}$：

$$\hat{\mathbf{H}}[g, p, :] = \mathbf{H}[g \cdot P + p, :], \quad g = 0, \ldots, \frac{N}{P}-1, \quad p = 0, \ldots, P-1$$

则 Split 等价于取第 $r$ 个切片：

$$\text{Split}_r(\mathbf{H}) = \hat{\mathbf{H}}[:, r, :] \in \mathbb{R}^{(N/P) \times D}$$

##### 2.3 数值示例

取 $N=64$，$P=8$。重塑后 $\hat{\mathbf{H}} \in \mathbb{R}^{8 \times 8 \times D}$：

$$\hat{\mathbf{H}} = \begin{pmatrix}
\mathbf{h}_0 & \mathbf{h}_1 & \mathbf{h}_2 & \cdots & \mathbf{h}_7 \\
\mathbf{h}_8 & \mathbf{h}_9 & \mathbf{h}_{10} & \cdots & \mathbf{h}_{15} \\
\mathbf{h}_{16} & \mathbf{h}_{17} & \mathbf{h}_{18} & \cdots & \mathbf{h}_{23} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\mathbf{h}_{56} & \mathbf{h}_{57} & \mathbf{h}_{58} & \cdots & \mathbf{h}_{63}
\end{pmatrix}$$

各 rank 取对应列：

- rank 0：$\text{Split}_0(\mathbf{H}) = [\mathbf{h}_0, \mathbf{h}_8, \mathbf{h}_{16}, \mathbf{h}_{24}, \mathbf{h}_{32}, \mathbf{h}_{40}, \mathbf{h}_{48}, \mathbf{h}_{56}]^\top$
- rank 3：$\text{Split}_3(\mathbf{H}) = [\mathbf{h}_3, \mathbf{h}_{11}, \mathbf{h}_{19}, \mathbf{h}_{27}, \mathbf{h}_{35}, \mathbf{h}_{43}, \mathbf{h}_{51}, \mathbf{h}_{59}]^\top$
- rank 7：$\text{Split}_7(\mathbf{H}) = [\mathbf{h}_7, \mathbf{h}_{15}, \mathbf{h}_{23}, \mathbf{h}_{31}, \mathbf{h}_{39}, \mathbf{h}_{47}, \mathbf{h}_{55}, \mathbf{h}_{63}]^\top$

**▶ V4 mHC 扩展**：V4 的 hidden\_states 为 $\mathbf{H} \in \mathbb{R}^{N \times M \times D}$，Split 算子在第 0 维操作，其余维度不变：

$$\text{Split}_r(\mathbf{H})[j, :, :] = \mathbf{H}[j \cdot P + r, :, :] \in \mathbb{R}^{M \times D}$$

对应 reshape 为 $\mathbb{R}^{(N/P) \times P \times M \times D}$ 后取第 $r$ 个切片。

---

#### 3. Position IDs 的 Split

对应 `cp_split_and_rebuild_position()`（[cp_utils.py:167](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L167)），在 round-robin-split 模式下同样调用 `dsa_cp_round_robin_split_data()`。

Position IDs 向量 $\mathbf{p} \in \mathbb{Z}^N$ 使用与 hidden\_states 完全相同的 Split 算子：

$$\text{Split}_r(\mathbf{p})[j] = \mathbf{p}[j \cdot P + r] = j \cdot P + r$$

即 rank $r$ 上的 position ID 序列为 $\{r, \; r+P, \; r+2P, \; \ldots\}$。

> **重要**：Split 后 position ID 保持原始的**绝对值**（非重新从 0 编号），确保 RoPE（旋转位置编码）的正确性。例如 rank 0 上的 position 序列为 $\{0, 8, 16, 24, \ldots\}$，而非 $\{0, 1, 2, 3, \ldots\}$。

---

#### 4. KV AllGather + Rerange 算子

对应 `cp_all_gather_rerange_output()`（[cp_utils.py:310](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L310)）。

##### 4.1 问题

Self-Attention 要求每个 rank 的 Q 能看到**完整的 $N$ 个 KV token**。Split 后每个 rank 只有 $N/P$ 个 KV，需要通过 AllGather 收集所有 rank 的 KV，然后**恢复原始 token 顺序**。

##### 4.2 AllGather 操作

设 rank $r$ 持有 $\mathbf{K}^{(r)} \in \mathbb{R}^{(N/P) \times D_{kv}}$。AllGather（NCCL `all_gather_into_tensor`）将所有 rank 的数据按 rank 编号顺序拼接：

$$\text{AllGather}(\mathbf{K}^{(0)}, \ldots, \mathbf{K}^{(P-1)}) = \text{concat}(\mathbf{K}^{(0)}, \mathbf{K}^{(1)}, \ldots, \mathbf{K}^{(P-1)}) \in \mathbb{R}^{N \times D_{kv}}$$

设结果为 $\mathbf{G} \in \mathbb{R}^{N \times D_{kv}}$。AllGather 后的 token 顺序是**交错的**——不是原始的 $0, 1, 2, \ldots, N-1$。具体地：

$$\mathbf{G}[r \cdot \tfrac{N}{P} + j, :] = \mathbf{K}^{(r)}[j, :] = \mathbf{K}_{\text{orig}}[j \cdot P + r, :]$$

展开来看，$\mathbf{G}$ 的行排列为：

$$\underbrace{\mathbf{h}_0, \mathbf{h}_P, \mathbf{h}_{2P}, \ldots}_{\text{rank 0 的 tokens}}, \; \underbrace{\mathbf{h}_1, \mathbf{h}_{P+1}, \mathbf{h}_{2P+1}, \ldots}_{\text{rank 1 的 tokens}}, \; \ldots, \; \underbrace{\mathbf{h}_{P-1}, \mathbf{h}_{2P-1}, \ldots}_{\text{rank } P{-}1 \text{ 的 tokens}}$$

##### 4.3 Rerange 操作

**目标**：将 $\mathbf{G}$ 恢复为原始顺序 $\mathbf{h}_0, \mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_{N-1}$。

Rerange 分三步完成：

**Step 1**：$\text{reshape}(P, \; N/P, \; D_{kv})$

将 $\mathbf{G} \in \mathbb{R}^{N \times D_{kv}}$ 重塑为 $\mathbf{G}_{\text{3D}} \in \mathbb{R}^{P \times (N/P) \times D_{kv}}$，按 rank 分块：

$$\mathbf{G}_{\text{3D}}[r, j, :] = \mathbf{G}[r \cdot \tfrac{N}{P} + j, :] = \mathbf{K}_{\text{orig}}[j \cdot P + r, :]$$

矩阵形式（行为 rank 编号 $r$，列为组内 token 编号 $j$）：

$$\mathbf{G}_{\text{3D}} = \begin{pmatrix}
\mathbf{h}_0 & \mathbf{h}_P & \mathbf{h}_{2P} & \cdots & \mathbf{h}_{(N/P-1)P} \\
\mathbf{h}_1 & \mathbf{h}_{P+1} & \mathbf{h}_{2P+1} & \cdots & \mathbf{h}_{(N/P-1)P+1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\mathbf{h}_{P-1} & \mathbf{h}_{2P-1} & \mathbf{h}_{3P-1} & \cdots & \mathbf{h}_{N-1}
\end{pmatrix}$$

**Step 2**：$\text{transpose}(0, 1)$

交换第 0、1 维，得到 $\mathbf{G}_{\text{T}} \in \mathbb{R}^{(N/P) \times P \times D_{kv}}$：

$$\mathbf{G}_{\text{T}}[j, r, :] = \mathbf{G}_{\text{3D}}[r, j, :] = \mathbf{K}_{\text{orig}}[j \cdot P + r, :]$$

矩阵形式（行为 token 组 $j$，列为 rank $r$）：

$$\mathbf{G}_{\text{T}} = \begin{pmatrix}
\mathbf{h}_0 & \mathbf{h}_1 & \mathbf{h}_2 & \cdots & \mathbf{h}_{P-1} \\
\mathbf{h}_P & \mathbf{h}_{P+1} & \mathbf{h}_{P+2} & \cdots & \mathbf{h}_{2P-1} \\
\mathbf{h}_{2P} & \mathbf{h}_{2P+1} & \mathbf{h}_{2P+2} & \cdots & \mathbf{h}_{3P-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\mathbf{h}_{(N/P-1)P} & \mathbf{h}_{(N/P-1)P+1} & \cdots & \cdots & \mathbf{h}_{N-1}
\end{pmatrix}$$

**Step 3**：$\text{reshape}(N, \; D_{kv})$

按行优先展平。对任意下标 $i \in \{0, \ldots, N-1\}$：

$$\text{Rerange}(\mathbf{G})[i, :] = \mathbf{G}_{\text{T}}[\lfloor i/P \rfloor, \; i \bmod P, :] = \mathbf{K}_{\text{orig}}[\lfloor i/P \rfloor \cdot P + i \bmod P, :] = \mathbf{K}_{\text{orig}}[i, :]$$

**结论**：$\text{Rerange}(\mathbf{G})[i, :] = \mathbf{K}_{\text{orig}}[i, :]$，Rerange 后恢复原始 token 顺序。

##### 4.4 合并公式

定义 **AllGather-Rerange 算子** $\text{Gather}$ 为 Split 的逆操作：

$$\text{Gather}(\mathbf{K}^{(0)}, \ldots, \mathbf{K}^{(P-1)}) \triangleq \text{Rerange}(\text{AllGather}(\mathbf{K}^{(0)}, \ldots, \mathbf{K}^{(P-1)}))$$

**性质（Split-Gather 互逆）**：

$$\text{Gather}(\text{Split}_0(\mathbf{H}), \; \text{Split}_1(\mathbf{H}), \; \ldots, \; \text{Split}_{P-1}(\mathbf{H})) = \mathbf{H}$$

##### 4.5 数值示例

$N=64, \; P=8$。

AllGather 后 $\mathbf{G}$ 的行排列（64 行，交错顺序）：

$$\mathbf{G} = [\underbrace{\mathbf{h}_0, \mathbf{h}_8, \ldots, \mathbf{h}_{56}}_{\text{rank 0}}, \; \underbrace{\mathbf{h}_1, \mathbf{h}_9, \ldots, \mathbf{h}_{57}}_{\text{rank 1}}, \; \ldots, \; \underbrace{\mathbf{h}_7, \mathbf{h}_{15}, \ldots, \mathbf{h}_{63}}_{\text{rank 7}}]$$

$\text{reshape}(8, 8, D_{kv})$ 后 $\mathbf{G}_{\text{3D}}$：

$$\mathbf{G}_{\text{3D}} = \begin{pmatrix}
\mathbf{h}_0 & \mathbf{h}_8 & \mathbf{h}_{16} & \mathbf{h}_{24} & \mathbf{h}_{32} & \mathbf{h}_{40} & \mathbf{h}_{48} & \mathbf{h}_{56} \\
\mathbf{h}_1 & \mathbf{h}_9 & \mathbf{h}_{17} & \mathbf{h}_{25} & \mathbf{h}_{33} & \mathbf{h}_{41} & \mathbf{h}_{49} & \mathbf{h}_{57} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\mathbf{h}_7 & \mathbf{h}_{15} & \mathbf{h}_{23} & \mathbf{h}_{31} & \mathbf{h}_{39} & \mathbf{h}_{47} & \mathbf{h}_{55} & \mathbf{h}_{63}
\end{pmatrix}$$

$\text{transpose}(0,1)$ 后 $\mathbf{G}_{\text{T}}$：

$$\mathbf{G}_{\text{T}} = \begin{pmatrix}
\mathbf{h}_0 & \mathbf{h}_1 & \mathbf{h}_2 & \mathbf{h}_3 & \mathbf{h}_4 & \mathbf{h}_5 & \mathbf{h}_6 & \mathbf{h}_7 \\
\mathbf{h}_8 & \mathbf{h}_9 & \mathbf{h}_{10} & \mathbf{h}_{11} & \mathbf{h}_{12} & \mathbf{h}_{13} & \mathbf{h}_{14} & \mathbf{h}_{15} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\mathbf{h}_{56} & \mathbf{h}_{57} & \mathbf{h}_{58} & \mathbf{h}_{59} & \mathbf{h}_{60} & \mathbf{h}_{61} & \mathbf{h}_{62} & \mathbf{h}_{63}
\end{pmatrix}$$

$\text{reshape}(64, D_{kv})$ 后：$[\mathbf{h}_0, \mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_{63}]$——恢复原始顺序。

---

#### 5. Input IDs Round-Robin（仅 V4 路径）

对应 `cp_round_robin_input_ids()`（[cp_utils.py:191](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/utils/cp_utils.py#L191)）。该操作有两个分支，取决于 MoE 的 a2a 后端。

##### 5.1 非 a2a 模式

设 $\mathbf{x} = [x_0, x_1, \ldots, x_{N-1}] \in \mathbb{Z}^N$ 为原始 input\_ids。

**Step 1**：$\text{reshape}(N/P, \; P)$

$$\mathbf{X} = \begin{pmatrix}
x_0 & x_1 & \cdots & x_{P-1} \\
x_P & x_{P+1} & \cdots & x_{2P-1} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N-P} & x_{N-P+1} & \cdots & x_{N-1}
\end{pmatrix} \in \mathbb{Z}^{(N/P) \times P}$$

**Step 2**：转置

$$\mathbf{X}^\top = \begin{pmatrix}
x_0 & x_P & x_{2P} & \cdots \\
x_1 & x_{P+1} & x_{2P+1} & \cdots \\
\vdots & \vdots & \vdots & \ddots \\
x_{P-1} & x_{2P-1} & x_{3P-1} & \cdots
\end{pmatrix} \in \mathbb{Z}^{P \times (N/P)}$$

**Step 3**：按行展平

$$\mathbf{x}' = [x_0, x_P, x_{2P}, \ldots, x_1, x_{P+1}, x_{2P+1}, \ldots, x_{P-1}, x_{2P-1}, \ldots, x_{N-1}]$$

**所有 rank 得到相同的 $\mathbf{x}'$**（完整 $N$ 个 token），用于 MoE routing。

> 注意：非 a2a 模式下 input\_ids 保持完整（$N$ 个），而 hidden\_states 已被 Split 为 $N/P$ 个。两者数量不对齐——input\_ids 仅作 MoE routing 索引，不与 hidden\_states 逐元素对应。

##### 5.2 a2a 模式（DeepEP）

每个 rank 做 stride 切片，与 hidden\_states 的 Split 算子完全相同：

$$\text{Split}_r(\mathbf{x})[j] = x_{j \cdot P + r}$$

每个 rank 只拿到 $N/P$ 个 token ID。

**▶ V2 路径差异**：V2 路径不做 input\_ids 的 round-robin 操作，也没有 `input_ids_global` 变量。MoE 的 expert routing 依赖 hidden\_states 本身而非 input\_ids。

---

#### 6. Self-Attention 的数学表达

对应 V4 的 `MQALayer._forward_prepare()`（[deepseek_v4.py:698](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L698)）和 V2 的 `DeepseekV2AttentionMLA` forward 路径。

##### 6.1 CP 模式下的 TP 约束

CP 启用时，Attention 层的 Tensor Parallelism 被强制设为 1（[deepseek_v4.py:258-259](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L258-L259)）。即每个 GPU 拥有全部 attention heads 权重的副本，独立完成完整的 attention 计算，不需要 TP 通信。

##### 6.2 每个 rank 的计算

在每一层 Transformer 中，rank $r$ 的 Self-Attention 计算如下。

**输入**：$\mathbf{H}^{(r)} = \text{Split}_r(\mathbf{H}) \in \mathbb{R}^{(N/P) \times D}$

**Q 计算**（每个 rank 独立计算自己的 $N/P$ 个 token 的 Q）：

$$\mathbf{Q}_r = f_Q(\mathbf{H}^{(r)}) \in \mathbb{R}^{(N/P) \times n_h \times d_h}$$

其中 $f_Q$ 包含线性投影（含 LoRA 的 down/up projection）、QK Norm、RoPE 等操作。

**KV 计算 + AllGather**：

1. 每个 rank 用自己的 $N/P$ 个 token 计算局部 KV：

$$\mathbf{K}^{(r)}, \mathbf{V}^{(r)} = f_{KV}(\mathbf{H}^{(r)}) \in \mathbb{R}^{(N/P) \times D_{kv}}$$

2. AllGather + Rerange 恢复完整的 KV（第 4 节的 $\text{Gather}$ 算子）：

$$\mathbf{K} = \text{Gather}(\mathbf{K}^{(0)}, \ldots, \mathbf{K}^{(P-1)}) \in \mathbb{R}^{N \times D_{kv}}$$
$$\mathbf{V} = \text{Gather}(\mathbf{V}^{(0)}, \ldots, \mathbf{V}^{(P-1)}) \in \mathbb{R}^{N \times D_{kv}}$$

**Attention 计算**：rank $r$ 用 $N/P$ 个 Q 对 $N$ 个 KV 做 attention（DSA 后端中为稀疏 attention）：

$$\mathbf{A}^{(r)} = \text{Attn}(\mathbf{Q}_r, \mathbf{K}, \mathbf{V}) \in \mathbb{R}^{(N/P) \times D}$$

> **关键**：Q 只有 $N/P$ 个，但 KV 是完整的 $N$ 个。每个 rank 计算一部分 Q 的 attention，但需要看到所有 KV。

##### 6.3 Attention 输出保持 split 状态

Attention 输出 $\mathbf{A}^{(r)} \in \mathbb{R}^{(N/P) \times D}$ 仍然是 split 状态，不立即 all-gather。各 rank 持有的 token 集合与 Split 后一致。

---

#### 7. MLP 层的通信数学（非 a2a 模式）

对应 `dsa_cp_gather_hidden_states()`（[communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55)）和 `dsa_cp_reduce_scatter_hidden_states()`（[communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67)）。

MLP/MoE 是 per-token 操作，但由于 Expert Parallelism (EP)，各 rank 持有不同的 experts 子集，需要看到完整的 token 集合才能完成 routing 和 expert 计算。因此 MLP 前后需要通信。

##### 7.1 MLP 前 AllGather（无 Rerange）

设 rank $r$ 输入为 $\mathbf{H}^{(r)} \in \mathbb{R}^{(N/P) \times D}$。AllGather 操作：

$$\tilde{\mathbf{H}} = \text{concat}(\mathbf{H}^{(0)}, \mathbf{H}^{(1)}, \ldots, \mathbf{H}^{(P-1)}) \in \mathbb{R}^{N \times D}$$

注意：**此处没有 Rerange**。$\tilde{\mathbf{H}}$ 的 token 顺序为：

$$\tilde{\mathbf{H}} = [\underbrace{\mathbf{h}_0, \mathbf{h}_P, \ldots}_{\text{rank 0}}, \; \underbrace{\mathbf{h}_1, \mathbf{h}_{P+1}, \ldots}_{\text{rank 1}}, \; \ldots]$$

**为什么可以不做 Rerange**：MLP（包括 MoE 的 per-token expert routing 和 FFN）是 **per-token 操作**——对每个 token 独立计算，不依赖 token 之间的相对顺序。形式化地：

$$\text{MLP}(\tilde{\mathbf{H}})[i, :] = f_{\text{MLP}}(\tilde{\mathbf{H}}[i, :])$$

其中 $f_{\text{MLP}}$ 仅依赖单个 token 的向量。设 $\sigma$ 为任意排列，则 $\text{MLP}(\sigma(\mathbf{H})) = \sigma(\text{MLP}(\mathbf{H}))$。因此 token 的排列顺序不影响最终计算结果，节省了一次 Rerange 操作。

##### 7.2 MoE EP Partial Result

在 Expert Parallelism 下，各 rank 持有不同的 expert 子集。设全部 $E$ 个 experts 中 rank $r$ 持有子集 $\mathcal{E}_r$。对于 token $i$，其 MoE 输出为所选 top-k expert 的加权和：

$$\text{MoE}(\mathbf{h}_i) = \sum_{e \in \text{TopK}(\mathbf{h}_i)} g_e(\mathbf{h}_i) \cdot \text{Expert}_e(\mathbf{h}_i)$$

rank $r$ 只能计算 $\mathcal{E}_r$ 中 expert 的贡献，输出为 **partial result**：

$$\mathbf{Y}_r[i, :] = \sum_{e \in \text{TopK}(\mathbf{h}_i) \cap \mathcal{E}_r} g_e(\mathbf{h}_i) \cdot \text{Expert}_e(\mathbf{h}_i)$$

完整的 MoE 输出为所有 rank 的 partial result 之和：

$$\mathbf{Y}_{\text{full}} = \sum_{r=0}^{P-1} \mathbf{Y}_r$$

> CP 启用时 `use_reduce_scatter=True`（[deepseek_v4.py:1420](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1420)），使 `should_skip_post_experts_all_reduce()`（[moe/utils.py:446](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/utils.py#L446)）返回 `True`，跳过 MoE 内部的 post-experts AllReduce。这一聚合操作被推迟到后续的 ReduceScatter 中合并完成。

##### 7.3 MLP 后 ReduceScatter

ReduceScatter 同时完成两件事：**跨 rank 求和**（reduce）+ **按 rank 分割**（scatter）。

设第 $r$ 个 rank 的 MLP 输出为 $\mathbf{Y}_r \in \mathbb{R}^{N \times D}$。将其等分为 $P$ 块：

$$\mathbf{Y}_r = [\mathbf{Y}_r^{[0]}, \; \mathbf{Y}_r^{[1]}, \; \ldots, \; \mathbf{Y}_r^{[P-1]}], \quad \mathbf{Y}_r^{[k]} \in \mathbb{R}^{(N/P) \times D}$$

ReduceScatter 后 rank $r$ 得到：

$$\mathbf{Z}^{(r)} = \sum_{k=0}^{P-1} \mathbf{Y}_k^{[r]} \in \mathbb{R}^{(N/P) \times D}$$

即 **buffer 中第 $r$ 块**在**所有 rank 上的 partial result 之和**。

**关键观察**：由于 AllGather 时没有 Rerange，buffer 中第 $r$ 块对应的正好是 rank $r$ 原来 Split 得到的那组 stride token。因此 ReduceScatter 后，rank $r$ 得到的是自己负责的那组 stride token 的完整 MLP 输出（EP partial results 已求和），回到了 split 状态。

##### 7.4 MLP 通信的完整数学表达

对于 rank $r$，一次 MLP 层的通信-计算-通信流程为：

$$\mathbf{H}^{(r)}_{\text{in}} \xrightarrow{\text{AllGather（无 rerange）}} \tilde{\mathbf{H}} \xrightarrow{\text{MLP}_r} \mathbf{Y}_r \xrightarrow{\text{ReduceScatter}} \mathbf{H}^{(r)}_{\text{out}}$$

其中：

$$\mathbf{H}^{(r)}_{\text{out}} = \text{ReduceScatter}_r\bigl(\text{MLP}_0(\tilde{\mathbf{H}}), \; \ldots, \; \text{MLP}_{P-1}(\tilde{\mathbf{H}})\bigr)$$

**▶ V2 路径差异**：V2 通过 `DSACPLayerCommunicator`（[communicator_dsa_cp.py:79](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L79)）的 `prepare_mlp()` 和 `postprocess_layer()` 隐式处理上述通信，而 V4 显式调用 `dsa_cp_gather_hidden_states()` / `dsa_cp_reduce_scatter_hidden_states()`。数学上等价。

##### 7.5 MLP AllGather 的必要性与跳过条件

**核心问题**：为什么 Round-Robin Split 模式下 MLP 前默认需要 AllGather？

**默认配置下 EP 并未开启**。Round-Robin Split 默认 `ep_size=1`（[server_args.py:635](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L635)），`moe_a2a_backend="none"`（[server_args.py:636](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L636)），且 round-robin-split 的初始化路径（[server_args.py:1870-1875](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1870-L1875)）**不会自动修改这两个值**。对比 In-Sequence Split 会自动设置 `ep_size=tp_size` 和 `moe_a2a_backend="deepep"`（[server_args.py:1865-1866](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/server_args.py#L1865-L1866)）。

**AllGather 的真正原因是 MoE 的 TP 权重切分**。`ep_size=1` 时，MoE 的 TP 大小计算如下：

$$P_{\text{moe\_tp}} = \frac{P_{\text{tp}}}{P_{\text{ep}} \times P_{\text{moe\_dp}}} = \frac{P}{1 \times 1} = P$$

即 `moe_tp_size = tp_size`（[distributed/parallel_state.py:2003](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L2003)）。这意味着每个 rank 虽然持有**全部 $E$ 个 expert**，但每个 expert 的权重 $\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3$ 在 intermediate_size 维度按 TP 切分，每个 rank 只持有 $1/P$ 的权重切片。

设 expert $e$ 的完整 FFN 权重为 $\mathbf{W}_e$，rank $r$ 持有的切片为 $\mathbf{W}_e^{(r)}$。对输入 token $\mathbf{h}_i$，rank $r$ 计算的是 **TP partial result**：

$$\text{Expert}_e^{(r)}(\mathbf{h}_i) = f(\mathbf{h}_i; \mathbf{W}_e^{(r)}) \neq \text{Expert}_e(\mathbf{h}_i)$$

完整结果需要跨 rank 求和：

$$\text{Expert}_e(\mathbf{h}_i) = \sum_{r=0}^{P-1} \text{Expert}_e^{(r)}(\mathbf{h}_i)$$

**正常（非 CP）流程**中，MoE 在前向结束时执行 TP AllReduce 合并 partial results。

**CP 流程**中，AllGather + ReduceScatter **替代**了 MoE 内部的 TP AllReduce。具体地，CP 传入 `use_reduce_scatter=True`，使 MoE 跳过内部的 TP AllReduce。随后 CP 的 ReduceScatter **同时完成两件事**：

$$\text{ReduceScatter} = \underbrace{\text{Reduce（跨 rank 求和 TP partial results）}}_{\text{替代 MoE 的 TP AllReduce}} + \underbrace{\text{Scatter（按 rank 分割回 split 状态）}}_{\text{恢复 CP 的 split 状态}}$$

因此 CP 的通信模式为：

$$\mathbf{H}^{(r)} \xrightarrow[\text{N/P → N}]{\text{AllGather}} \tilde{\mathbf{H}} \xrightarrow{\text{MoE}_r(\text{TP partial})} \mathbf{Y}_r \xrightarrow[\text{reduce + scatter}]{\text{ReduceScatter}} \mathbf{H}^{(r)}_{\text{out}}$$

**跳过条件——All-to-All 后端 + EP**：当使用 `--moe-a2a-backend deepep --ep-size <tp_size>` 时，EP 模式开启，TP 大小变为：

$$P_{\text{moe\_tp}} = \frac{P}{P \times 1} = 1$$

此时每个 expert 的权重**不再做 TP 切分**——每个 rank 持有 $E/P$ 个 expert 的完整权重。Token 的跨 rank dispatch 由 DeepEP all-to-all 处理：

$$\text{DeepEP}: \quad \mathbf{H}^{(r)} \xrightarrow{\text{a2a dispatch}} \{\mathbf{h}_i \mid \text{TopK}(\mathbf{h}_i) \cap \mathcal{E}_r \neq \varnothing\} \xrightarrow{\text{Expert}_r} \xrightarrow{\text{a2a collect}} \mathbf{Z}^{(r)}$$

a2a 后端将 AllGather + MLP + ReduceScatter 三步合并为一个自洽的 dispatch-compute-collect 流水线，**无需 CP 层面的额外通信**。

**控制参数**：`moe_a2a_backend` 和 `ep_size` 共同决定使用哪种模式。

| 配置 | `ep_size` | `moe_tp_size` | Expert 权重 | MLP 前 AllGather | MLP 后 ReduceScatter | 单层 CP 通信 |
|---|---|---|---|---|---|---|
| 默认（non-a2a） | 1 | $P$ | TP 切分（$1/P$ 权重） | 需要 | 需要（替代 TP AllReduce） | 3 次 |
| a2a（DeepEP） | $P$ | 1 | EP 切分（$E/P$ 完整 expert） | 跳过 | 跳过 | 1 次 |

> Round-Robin Split 模式不会自动设置 `moe_a2a_backend` 和 `ep_size`，因此**默认走 non-a2a 路径**（`ep_size=1`，TP 权重切分），每层需要 3 次通信。如需跳过 MLP AllGather/ReduceScatter，须显式指定 `--moe-a2a-backend deepep --ep-size <tp_size>`，切换到 EP 模式。相比之下，In-Sequence Split 模式会自动强制两者。

**通信量对比**：对于 $L$ 层模型，总 CP 通信次数为：

$$C_{\text{non-a2a}} = 3L, \qquad C_{\text{a2a}} = L$$

a2a + EP 模式将通信次数降低至 $1/3$。

---

#### 8. 单层 Transformer Block 的完整数学描述

对第 $l$ 层，rank $r$ 的计算流程（以 V2 路径为例，V4 多 mHC 维度但原理相同）：

**输入**：$\mathbf{H}^{(r)} \in \mathbb{R}^{(N/P) \times D}$（split 状态）

**1. Self-Attention**（1 次 AllGather 通信）

$$\mathbf{Q}_r = f_Q^{(l)}(\mathbf{H}^{(r)})$$

$$\mathbf{K}^{(r)}, \mathbf{V}^{(r)} = f_{KV}^{(l)}(\mathbf{H}^{(r)})$$

$$\mathbf{K} = \text{Gather}(\mathbf{K}^{(0)}, \ldots, \mathbf{K}^{(P-1)}) \qquad \leftarrow \text{通信 1: KV AllGather + Rerange}$$

$$\mathbf{A}^{(r)} = \text{Attn}(\mathbf{Q}_r, \mathbf{K}, \mathbf{V}) \in \mathbb{R}^{(N/P) \times D} \qquad \text{（仍为 split 状态）}$$

**2. MLP**（2 次通信：AllGather + ReduceScatter）

$$\tilde{\mathbf{A}} = \text{AllGather}_{\text{no-rerange}}(\mathbf{A}^{(0)}, \ldots, \mathbf{A}^{(P-1)}) \in \mathbb{R}^{N \times D} \qquad \leftarrow \text{通信 2: MLP AllGather}$$

$$\mathbf{Y}_r = \text{MLP}_r^{(l)}(\tilde{\mathbf{A}}) \in \mathbb{R}^{N \times D} \qquad \text{（TP partial result，默认 ep\_size=1）}$$

$$\mathbf{H}^{(r)}_{\text{out}} = \text{ReduceScatter}_r(\mathbf{Y}_0, \ldots, \mathbf{Y}_{P-1}) \in \mathbb{R}^{(N/P) \times D} \qquad \leftarrow \text{通信 3: ReduceScatter}$$

**输出**：$\mathbf{H}^{(r)}_{\text{out}}$（split 状态，传入第 $l+1$ 层）

##### 每层通信次数

| 通信操作 | 次数 | 目的 |
|----------|------|------|
| KV AllGather + Rerange | 1 | 收集完整 KV，恢复 token 顺序 |
| MLP AllGather（无 Rerange） | 1 | 收集完整 hidden\_states 给 MLP |
| MLP ReduceScatter | 1 | 聚合 EP partial result + 恢复 split |
| **合计** | **3** | — |

---

#### 9. Q 序列长度分配公式

对应 `dsa_cp_round_robin_split_q_seqs_cpu()`（[dsa/utils.py:221](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/attention/dsa/utils.py#L221)）。

当 batch 中有多个请求时，需要计算每个请求在每个 rank 上分配到的 Q token 数。

##### 9.1 单请求情形

对于单个长度为 $L$ 的请求（$L \bmod P = 0$），rank $r$ 分到的 Q token 数为：

$$q_r = \frac{L}{P}$$

所有 rank 平均分配。

##### 9.2 多请求情形（带余量累积）

对于 batch 中第 $b$ 个请求（$b = 0, 1, \ldots, B-1$），extend 长度为 $L_b$。

设累积余量 $e_0 = 0$，有效长度 $L'_b = L_b + e_b$，则 rank $r$ 在第 $b$ 个请求上分到的 Q token 数为：

$$q_r^{(b)} = \left\lfloor \frac{L'_b}{P} \right\rfloor + \mathbb{1}[L'_b \bmod P > r]$$

其中 $\mathbb{1}[\cdot]$ 为指示函数（条件为真时值为 1，否则为 0）。余量更新：

$$e_{b+1} = L'_b - q_r^{(b)} \cdot P$$

**直觉**：$L'_b / P$ 的整数部分平均分给所有 rank；余数 $L'_b \bmod P$ 个 token 分给编号最小的若干 rank（rank $r < L'_b \bmod P$ 的多分 1 个）。未分完的余量累积到下一个请求。

##### 9.3 示例

**示例 1**：$P=8$，单请求 $L=64$

$$q_r = 64 / 8 = 8, \quad \forall r \in \{0, \ldots, 7\}$$

所有 rank 各处理 8 个 Q token。

**示例 2**：$P=8$，单请求 $L=70$

$$L' = 70, \quad \lfloor 70/8 \rfloor = 8, \quad 70 \bmod 8 = 6$$

| rank | $q_r$ | 计算 |
|------|-------|------|
| 0~5 | 9 | $8 + \mathbb{1}[6 > r] = 8 + 1 = 9$（$r < 6$） |
| 6~7 | 8 | $8 + \mathbb{1}[6 > r] = 8 + 0 = 8$（$r \geq 6$） |

验证：$6 \times 9 + 2 \times 8 = 54 + 16 = 70$ ✓

---

#### 10. 最后一层的恢复

经过所有 $L$ 层 Transformer 后，最后一层输出的 $\mathbf{H}^{(r)} \in \mathbb{R}^{(N/P) \times D}$ 需要恢复为完整序列用于 logits 计算。

对应 [deepseek_v4.py:1615-1621](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1615-L1621)（V4）和 [deepseek_v2.py:2458-2472](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L2458-L2472)（V2）。

##### 10.1 V4 路径：AllGather → hc_head → norm

$$\mathbf{H}_{\text{full}} = \text{Gather}(\mathbf{H}^{(0)}, \ldots, \mathbf{H}^{(P-1)}) \in \mathbb{R}^{N \times M \times D}$$

$$\mathbf{H}_{\text{flat}} = \text{flatten}_{M}(\mathbf{H}_{\text{full}}) \in \mathbb{R}^{N \times (M \cdot D)}$$

$$\mathbf{H}_{\text{hc}} = f_{\text{hc\_head}}(\mathbf{H}_{\text{flat}}) \in \mathbb{R}^{N \times D}$$

$$\mathbf{H}_{\text{norm}} = \text{RMSNorm}(\mathbf{H}_{\text{hc}}) \in \mathbb{R}^{N \times D}$$

其中 $f_{\text{hc\_head}}$ 是 mHC 压缩操作（将 $M$ 个 head 的 composition 映射回 $D$ 维），$\text{flatten}_M$ 将第 1、2 维展平。

##### 10.2 V2 路径：norm → AllGather

$$\mathbf{H}_{\text{norm}}^{(r)} = \text{RMSNorm}(\mathbf{H}^{(r)}) \in \mathbb{R}^{(N/P) \times D}$$

$$\mathbf{H}_{\text{full}} = \text{Gather}(\mathbf{H}_{\text{norm}}^{(0)}, \ldots, \mathbf{H}_{\text{norm}}^{(P-1)}) \in \mathbb{R}^{N \times D}$$

##### 10.3 V2 "norm → gather" 与 V4 "gather → norm" 的数学等价性

RMSNorm 定义为（$\boldsymbol{\gamma} \in \mathbb{R}^D$ 为可学习参数）：

$$\text{RMSNorm}(\mathbf{h}) = \frac{\mathbf{h}}{\sqrt{\frac{1}{D}\sum_{d=1}^{D} h_d^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

由于 RMSNorm 是 **per-token 操作**（只依赖单个 token 的向量，没有跨 token 依赖），对 split 状态的 token 分别做 norm 与对 gather 后的完整序列做 norm 结果相同：

$$\text{Gather}\bigl(\text{RMSNorm}(\mathbf{H}^{(0)}), \ldots, \text{RMSNorm}(\mathbf{H}^{(P-1)})\bigr) = \text{RMSNorm}\bigl(\text{Gather}(\mathbf{H}^{(0)}, \ldots, \mathbf{H}^{(P-1)})\bigr)$$

V2 先 norm 再 gather 的好处是 AllGather 传输的数据量为 $N \times D$（norm 后的维度），而 V4 传输 $N \times M \times D$（含 mHC 维度，norm 前），通信量是 V2 的 $M$ 倍。V4 不得不先 gather 的原因是 $f_{\text{hc\_head}}$ 操作需要完整序列的 `pre_hc_head` tensor 作为 `LogitsProcessor` 的额外输入。

---

#### 11. 通信复杂度分析

##### 11.1 单层通信量

设 bf16 下每个元素 2 字节，hidden dimension 为 $D$，KV 维度为 $D_{kv}$。

| 操作 | 每 rank 发送 | 每 rank 接收 |
|------|-------------|-------------|
| KV AllGather | $\frac{N}{P} \cdot D_{kv} \cdot 2$ 字节 | $N \cdot D_{kv} \cdot 2$ 字节 |
| MLP AllGather | $\frac{N}{P} \cdot D \cdot 2$ 字节 | $N \cdot D \cdot 2$ 字节 |
| MLP ReduceScatter | $N \cdot D \cdot 2$ 字节 | $\frac{N}{P} \cdot D \cdot 2$ 字节 |

##### 11.2 全模型通信量

设模型有 $L$ 层，单层通信总量（所有 rank 合计发送）为：

$$C_{\text{layer}} = N \cdot D_{kv} \cdot 2 + 2 \cdot N \cdot D \cdot 2 = 2N(D_{kv} + 2D) \text{ 字节}$$

全模型通信总量：

$$C_{\text{total}} = L \cdot C_{\text{layer}} + C_{\text{last}}$$

其中 $C_{\text{last}}$ 为最后一层的 AllGather 恢复完整序列：V2 为 $2ND$ 字节，V4 为 $2NMD$ 字节。

##### 11.3 CP 加速比上界

理想情况下（通信完全被计算重叠），Self-Attention 的计算复杂度从 $O(N^2)$ 降低到：

$$O\left(\frac{N}{P} \cdot N\right) = O\left(\frac{N^2}{P}\right)$$

因为每个 rank 只计算 $N/P$ 个 Q 对 $N$ 个 KV 的 attention。理论加速比上界为 $P$。

实际加速比低于 $P$，原因：

1. 每层 3 次集合通信的延迟无法完全被计算重叠
2. KV 仍需完整——每个 rank 存储 $N$ 个 KV，内存节省仅在 Q 侧（$N \to N/P$）
3. MLP 层无法加速——AllGather 后每个 rank 仍处理完整的 $N$ 个 token

---

#### 12. Split-Gather 对偶性总结

Round-Robin-Split CP 的核心数学结构是 **Split-Gather 对偶性**。定义四个基本算子：

| 算子 | 数学定义 | 作用 |
|------|----------|------|
| $\text{Split}_r$ | $\hat{\mathbf{H}}[:, r, :]$，其中 $\hat{\mathbf{H}} = \text{reshape}(\mathbf{H}, \; N/P, \; P, \; D)$ | 将完整序列分发给各 rank |
| $\text{Gather}$ | $\text{reshape}(\text{AllGather}, \; P, \; N/P, \; D) \to \text{transpose}(0,1) \to \text{reshape}(N, \; D)$ | 从各 rank 收集并恢复原始顺序 |
| $\text{AllGather}_{\text{nr}}$ | $\text{concat}(\mathbf{H}^{(0)}, \ldots, \mathbf{H}^{(P-1)})$（不做 rerange） | MLP 前收集（per-token 操作不需要正确顺序） |
| $\text{ReduceScatter}_r$ | $\sum_{k=0}^{P-1} \mathbf{Y}_k^{[r]}$ | 跨 rank 求和 + 重新分割 |

**核心性质**：

$$\text{Gather} \circ (\text{Split}_0, \ldots, \text{Split}_{P-1}) = \text{Id}$$

**不变量**：在 Transformer 的每一层，输入和输出都处于 split 状态，形成闭合循环：

$$\text{split} \xrightarrow[\text{KV AllGather+Rerange}]{\text{Self-Attention}} \text{split} \xrightarrow[\text{AllGather → MLP → ReduceScatter}]{\text{MLP/MoE}} \text{split}$$

---

#### 13. TP vs EP 通信本质差异：为何 token 整传

对应 `dsa_cp_gather_hidden_states()`（[communicator_dsa_cp.py:55](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L55)）和 `dsa_cp_reduce_scatter_hidden_states()`（[communicator_dsa_cp.py:67](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67)）。

##### 13.1 问题

默认 `ep_size=1` 时 MoE 的 `moe_tp_size=P`（权重在 intermediate_size 维度切 $P$ 份）。既然每卡只持 $1/P$ 权重切片，为何 MLP AllGather 时要传**完整**的 $N$ 个 token（而非每卡只传 $1/P$）？如果能切 token，TP 的通信量应与 EP 相当，但实测 EP 明显更快——原因何在？

##### 13.2 TP 模式：token 必须整传

TP 切的是**权重**，不是 token。每个 token 的完整 MoE 输出需要**所有 $P$ 张卡的权重切片 partial 求和**：

$$\text{Expert}_e(\mathbf{h}_i) = \sum_{r=0}^{P-1} \text{Expert}_e^{(r)}(\mathbf{h}_i)$$

因此每个 token $\mathbf{h}_i$ 必须**被所有 $P$ 张卡同时看到**——如果某张卡缺少 $\mathbf{h}_i$，就无法算出自己那份权重切片 partial，reduce 时缺一份，结果不完整。

AllGather 实现（[communicator_dsa_cp.py:59-63](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L59-L63)）：

```python
# MLP 前，每卡 local = N/P 个 token 的 hidden_states (shape: [N/P, D])
hidden_states, local_hidden_states = (
    get_local_dp_buffer(get_attention_cp_group()),
    hidden_states,
)
attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)
# AllGather 后，hidden_states = P 卡拼接 (shape: [N, D])
# 每卡发 N/P × D，收 N × D —— token 的 D 维完整保留，不切分
```

`all_gather_into_tensor` 是 NCCL AllGather：每卡把自己**完整** local 数据广播给所有卡，结果每卡得到所有卡数据的拼接。token 的 hidden state **不会被切分**——每卡拿到完整的 $[N, D]$，即全 $N$ 个 token、每个 token 的 $D$ 维完整保留。

ReduceScatter 同理（[communicator_dsa_cp.py:67-76](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/communicator_dsa_cp.py#L67-L76)）：每卡发 $N \times D$（全量 TP partial），收 $N/P \times D$。

##### 13.3 EP 模式：token 只发给所需专家所在卡

EP 切的是**专家**，不是权重。每卡持 $E/P$ 个 expert 的**完整权重**（无切片），token 只需发给其 TopK 选中的 expert 所在卡：

$$\text{rank } r \text{ 处理的 token 集合} = \{\mathbf{h}_i \mid \text{TopK}(\mathbf{h}_i) \cap \mathcal{E}_r \neq \varnothing\}$$

每个 token 的 TopK 专家分散在不同卡上，token 通过 a2a dispatch **点对点**发给每张相关卡，算完后 a2a combine **点对点**回原归属卡。不涉及 token 的全量广播。

##### 13.4 通信量对比

设 $N$ 个 token、hidden 维度 $D$、bf16（2 字节）、TopK = $K$、expert 总数 $E$、并行度 $P$。

| | TP=$P$ (AllGather+ReduceScatter) | EP=$P$ (a2a dispatch+combine) |
|---|---|---|
| MLP 前 | AllGather: 每卡收 $N \times D \times 2$ 字节（全量 token） | dispatch: 每卡只收路由到自己的 token 子集 |
| 计算 | 每卡对**全 $N$** token 算 $1/P$ 权重 partial | 每卡只算路由到自己的 token 子集（≈ $N \cdot K \cdot P / E$ 个） |
| MLP 后 | ReduceScatter: 每卡发 $N \times D \times 2$ 字节 | combine: 结果点对点回原归属卡 |
| 每卡通信量 | $\approx 2 \cdot N \cdot D \cdot 2$ 字节（dense 全量） | $\approx 2 \cdot (N \cdot K / E \cdot P) \cdot D \cdot 2$ 字节（sparse） |

关键差异：

- **TP 是 dense 通信**：所有卡必须看到全量 $N$ 个 token（因为权重切片求和需要），与路由无关。通信量与 token 选了哪些 expert 无关。
- **EP 是 sparse 通信**：token 只发给所需 expert 所在卡，未命中 expert 的卡不参与。通信量正比于实际路由 token 数，远小于全量。

以 DeepSeek V4 典型参数为例（$K=8$, $E=256$, $P=8$）：

$$\text{TP 每卡收发} \propto N \cdot D, \qquad \text{EP 每卡收发} \propto N \cdot \frac{K \cdot P}{E} \cdot D = N \cdot \frac{8 \times 8}{256} \cdot D = 0.25 \cdot N \cdot D$$

EP 通信量约为 TP 的 $1/4$。

##### 13.5 TP 计算冗余

除通信量更大外，TP 模式还有**计算冗余**：每张卡对全 $N$ 个 token 都计算自己那份权重切片 partial，但实际上不同 token 可能选完全不同的 expert。$P$ 张卡对同一 token 各算 $1/P$ 权重 partial 再求和——等价于 1 张卡算完整权重，但消耗了 $P$ 倍算力。

EP 模式无此冗余：每张卡只算路由到自己专家的 token，各卡计算天然不重叠。

| | TP=$P$ | EP=$P$ |
|---|---|---|
| 每 token 计算次数 | $P$ 次（$P$ 张卡各算 $1/P$ 权重 partial） | $K$ 次（$K$ 个被选中 expert 各算 1 次完整权重） |
| 总计算量 | $N \cdot E_{\text{active}} \cdot P \cdot \frac{1}{P} = N \cdot E_{\text{active}}$ | $N \cdot K$ |
| 算力效率 | 低（$P$ 份 partial 拼出 1 份完整结果） | 高（每份计算都产出完整结果） |

> 注：$E_{\text{active}}$ 为每个 token 实际激活的 expert 数（等于 $K$），TP 模式下每卡对全 $N$ 个 token 都要计算所有激活 expert 的 partial，而 EP 只计算路由到自己的 token 对应的 expert。

##### 13.6 代码流程对比（cp=$P$ 固定）

对应 [deepseek_v4.py:1399-1428](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v4.py#L1399-L1428)。

**默认 ep=1 (moe_tp=$P$)**：

```python
# MLP 前: 全量 AllGather (N/P → N)
if get_moe_a2a_backend().is_none():
    hidden_states = dsa_cp_gather_hidden_states(hidden_states)

# MoE: 每卡持全 E expert × 1/P 权重, 对全 N token 算 TP partial
hidden_states = self.mlp(hidden_states, ..., use_reduce_scatter=True)

# MLP 后: ReduceScatter (N → N/P, 替代 MoE TP AllReduce)
if get_moe_a2a_backend().is_none():
    hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
```

**deepep ep=$P$ (moe_tp=1)**：

```python
# MLP 前: 不做 AllGather! a2a dispatch 内化 token 分发
assert get_moe_a2a_backend().is_deepep()

# MoE: 每卡持 E/P 个完整 expert, a2a dispatch+compute+combine
hidden_states = self.mlp(hidden_states, ..., use_reduce_scatter=True)
# a2a 内部: dispatch(token→expert卡) → compute(完整权重) → combine(结果回原卡)

# MLP 后: 不做 ReduceScatter! a2a combine 已内化结果回收
```

**结论**：TP 模式 MLP 前后的两次 dense 集合通信（AllGather + ReduceScatter）在 EP 模式下被 DeepEP a2a 的 sparse 点对点通信取代，同时消除了 TP partial 的计算冗余。这是 EP 通信效率优于 TP 的根本原因——不是"省了"通信，而是换了更匹配 MoE 稀疏路由特性的通信载体。

---

#### 14. MoE 原理与 EP/TP 拆分机制

##### 14.1 MoE（Mixture of Experts）基本原理

MoE 层的核心思想：**不是每个 token 都经过全部 expert，而是由路由器（Router/Gate）为每个 token 选择最相关的 $K$ 个 expert（Top-K 路由）**，只计算被选中的 expert，再按路由权重加权求和。

对应 `DeepseekV2MoE`（[deepseek_v2.py:511](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L511)）。

**模型参数**（以 DeepSeek V3/V4 为例）：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| $E$ | 路由 expert 总数（`n_routed_experts`） | 256 |
| $K$ | 每个 token 选中的 expert 数（`num_experts_per_tok`） | 8 |
| $E_s$ | 共享 expert 数（`n_shared_experts`） | 1 |
| $D$ | hidden dimension | 7168 |
| $D_i$ | expert intermediate size（`moe_intermediate_size`） | 2048 |
| $\alpha$ | 路由缩放因子（`routed_scaling_factor`） | 2.5 |

**单 token 的 MoE 计算流程**：

设输入 token $\mathbf{h} \in \mathbb{R}^{D}$。

**Step 1: Router 计算**

Gate 权重 $\mathbf{W}_g \in \mathbb{R}^{E \times D}$（[deepseek_v2.py:418](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L418)），计算路由 logits：

$$\mathbf{l} = \mathbf{h} \cdot \mathbf{W}_g^\top \in \mathbb{R}^{E}$$

**Step 2: Top-K 选择**

从 $E$ 个 logits 中选取最大的 $K$ 个（V3 为 grouped top-k，V4 为 ungrouped），得到选中 expert 集合 $\mathcal{S} = \{e_1, e_2, \ldots, e_K\}$ 和对应路由权重 $g_{e_k}$（经 softmax 归一化）。

**Step 3: Expert 计算**

每个 expert 是一个标准 FFN（SwiGLU 激活），含三个权重矩阵：

$$\text{Expert}_e(\mathbf{h}) = \mathbf{W}_{\text{down},e} \cdot \text{SiLU}(\mathbf{h} \cdot \mathbf{W}_{\text{gate},e}^\top) \odot (\mathbf{h} \cdot \mathbf{W}_{\text{up},e}^\top)$$

其中 $\mathbf{W}_{\text{gate},e}, \mathbf{W}_{\text{up},e} \in \mathbb{R}^{D_i \times D}$，$\mathbf{W}_{\text{down},e} \in \mathbb{R}^{D \times D_i}$。

**Step 4: 加权求和**

$$\text{MoE}_{\text{routed}}(\mathbf{h}) = \alpha \cdot \sum_{k=1}^{K} g_{e_k} \cdot \text{Expert}_{e_k}(\mathbf{h})$$

**Step 5: 加上共享 expert**

共享 expert 对所有 token 都计算（不经路由）：

$$\text{MoE}(\mathbf{h}) = \text{MoE}_{\text{routed}}(\mathbf{h}) + \text{SharedExpert}(\mathbf{h})$$

**计算量分析**：

- 单 token 的 routed expert 计算：$K \times (2 \cdot D \cdot D_i + D_i \cdot D) = K \cdot 3 \cdot D \cdot D_i$ FLOPs
- 单 token 的共享 expert 计算：$E_s \cdot 3 \cdot D \cdot D_i$ FLOPs
- 与 dense FFN（$3 \cdot D \cdot D_{\text{dense}}$）的比值：$\frac{(K + E_s) \cdot D_i}{D_{\text{dense}}} \ll 1$——MoE 以远小于 dense 的计算量获得更大的参数空间

**关键特性**：每个 token 只激活 $K / E$ 的 expert 参数，这是 MoE 稀疏性的根源，也是 EP 拆分高效的基础。

##### 14.2 TP 拆分：按权重维度切分

对应 `FusedMoE.__init__`（[fused_moe_triton/layer.py:136](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L136)）和 [distributed/parallel_state.py:2059](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L2059)。

**核心思想**：每个 expert 的权重在 **intermediate_size 维度**上切分到 $P_{\text{moe\_tp}}$ 张卡，每张卡持 $1/P_{\text{moe\_tp}}$ 的权重切片，所有卡持**全量 $E$ 个 expert**。

**公式**（[fused_moe_triton/layer.py:215-216](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L215-L216)）：

$$D_{i,\text{per\_partition}} = \frac{D_i}{P_{\text{moe\_tp}}}$$

**权重分布**：

```
TP=4, E=256, Di=2048 时:

GPU0: expert_0..255 × [W_gate(512×D), W_up(512×D), W_down(D×512)]    ← 1/4 权重切片
GPU1: expert_0..255 × [W_gate(512×D), W_up(512×D), W_down(D×512)]    ← 1/4 权重切片
GPU2: expert_0..255 × [W_gate(512×D), W_up(512×D), W_down(D×512)]    ← 1/4 权重切片
GPU3: expert_0..255 × [W_gate(512×D), W_up(512×D), W_down(D×512)]    ← 1/4 权重切片

每卡持: E=256 个 expert, 但每个 expert 只有 1/4 的权重
```

**Gate/Router 权重**：$\mathbf{W}_g \in \mathbb{R}^{E \times D}$ **不分片**——每卡持完整 router 权重，独立计算 Top-K 路由（[deepseek_v2.py:472](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L472)）。

**TP 前向流程**（[deepseek_v2.py:927-1034](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L927-L1034) `forward_normal`）：

```
输入: hidden_states [N, D] (每卡持全 N 个 token)

1. Gate 计算:   logits = hidden_states @ W_g^T        → [N, E]     每卡独立
2. Top-K:       选中 expert 集合 + 路由权重             → [N, K]     每卡独立
3. Expert 计算: 每卡对全 N token, 用 1/P 权重算 partial  → [N, D]     每卡结果不同!
4. TP AllReduce: 跨卡求和 partial                      → [N, D]     得到完整结果
5. 加 SharedExpert (同理 TP 切分 + AllReduce)
```

**关键约束**：步骤 3 中每张卡必须看到**全 $N$ 个 token**，因为每个 token 的完整 expert 输出需要所有 $P$ 张卡的权重切片 partial 求和。如果某张卡缺少某个 token，则该 token 在该卡上的 partial 缺失，reduce 后结果不完整。

**TP 模式的冗余**：

| 维度 | 每卡持有 | 说明 |
|------|---------|------|
| Expert 数 | $E$（全量） | 所有卡持全部 expert |
| 权重量 | $E \times \frac{D_i}{P} \times D \times 3$ | 每个专家只有 $1/P$ 权重 |
| 计算量 | $N \times E_{\text{active}} \times \frac{1}{P} \times 3DD_i$ | 对全 $N$ token 算 $1/P$ partial |
| 通信 | TP AllReduce ($N \times D \times 2$ bytes) | dense 全连接 |
| 内存 | 总模型参数 $/ P$ | 跟 EP 相同 |

总计算量 = $P$ 卡 $\times$ $N$ token $\times$ $E_{\text{active}}$ $\times$ $\frac{1}{P}$ 权重 = $N \times E_{\text{active}} \times 3DD_i$，与单卡 dense 等价。即 TP 只切分内存，不减少总计算量——$P$ 份 partial 拼出 1 份完整结果。

##### 14.3 EP 拆分：按 expert 切分

对应 `FusedMoE.__init__`（[fused_moe_triton/layer.py:196-212](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L196-L212)）、`_map_global_expert_id_to_local_expert_id`（[fused_moe_triton/layer.py:590](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L590)）和 [distributed/parallel_state.py:2032](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L2032)。

**核心思想**：将 $E$ 个 expert **按编号分块**分配到 $P_{\text{moe\_ep}}$ 张卡，每张卡持 $E / P_{\text{moe\_ep}}$ 个 expert 的**完整权重**（不切片）。

**Expert 映射公式**（[fused_moe_triton/layer.py:590-598](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L590-L598)）：

rank $r$ 持有的 expert 编号范围：

$$\mathcal{E}_r = \left\{r \cdot \frac{E}{P}, \; r \cdot \frac{E}{P} + 1, \; \ldots, \; (r+1) \cdot \frac{E}{P} - 1\right\}$$

本地 expert ID 映射：

$$\text{local\_id}(e) = e - r \cdot \frac{E}{P}, \quad e \in \mathcal{E}_r$$

若 $e \notin \mathcal{E}_r$，返回 $-1$（该 expert 不在本卡，不加载其权重）。

**权重分布**：

```
EP=4, E=256, Di=2048 时:

GPU0: expert_0..63   × [W_gate(2048×D), W_up(2048×D), W_down(D×2048)]   ← 完整权重
GPU1: expert_64..127  × [W_gate(2048×D), W_up(2048×D), W_down(D×2048)]   ← 完整权重
GPU2: expert_128..191 × [W_gate(2048×D), W_up(2048×D), W_down(D×2048)]   ← 完整权重
GPU3: expert_192..255 × [W_gate(2048×D), W_up(2048×D), W_down(D×2048)]   ← 完整权重

每卡持: E/4 = 64 个 expert, 每个 expert 权重完整 (无切片)
```

**Gate/Router 权重**：$\mathbf{W}_g \in \mathbb{R}^{E \times D}$ 仍**不分片**——每卡持完整 router 权重，独立计算 Top-K 路由，确定每个 token 需要哪些 expert。

**EP 前向流程**（[deepseek_v2.py:1097](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/models/deepseek_v2.py#L1097) `forward_deepep`，DeepEP a2a 后端）：

```
输入: hidden_states [N, D] (每卡只持 N/P 个 token，CP split 后)

1. Gate 计算:    logits = hidden_states @ W_g^T       → [N/P, E]    每卡独立
2. Top-K:        选中 expert 集合 + 路由权重            → [N/P, K]    每卡独立
3. a2a dispatch: token 按其 TopK expert 路由到对应卡    → 每卡收到路由给自己的 token 子集
4. Expert 计算:  每卡对收到的 token 算完整 expert       → 无 partial!
5. a2a combine:  算完的 token 结果回原归属卡             → 每卡恢复自己的 N/P 个 token 结果
6. 加 SharedExpert (共享 expert 复制到每卡, TP1 模式)
```

**关键优势**：步骤 4 中每张卡持**完整 expert 权重**，计算结果直接完整，无需跨卡求和。不存在 TP partial 的概念。

**EP 模式的特性**：

| 维度 | 每卡持有 | 说明 |
|------|---------|------|
| Expert 数 | $E / P$ | 只持部分 expert |
| 权重量 | $\frac{E}{P} \times D_i \times D \times 3$ | 每个 expert 权重完整 |
| 计算量 | 只算路由到自己的 token | 无冗余，每份计算都有意义 |
| 通信 | a2a dispatch + combine (sparse) | 点对点，只传需要的 token |
| 内存 | 总模型参数 $/ P$ | 跟 TP 相同 |

##### 14.4 TP 与 EP 的本质对比

| 维度 | TP（权重切分） | EP（专家切分） |
|------|---------------|---------------|
| **切分对象** | expert 的权重矩阵 | expert 本身 |
| **每卡 expert 数** | $E$（全量） | $E / P$ |
| **每卡权重完整度** | $1/P$ 切片 | 完整 |
| **token 可见性** | 每卡须见全 $N$ token | 每卡只收路由到自己的 token |
| **计算结果** | TP partial（需跨卡求和） | 完整（无需求和） |
| **通信模式** | dense AllGather + ReduceScatter | sparse a2a dispatch + combine |
| **通信内容** | 全量 token hidden_states | 只传路由命中的 token |
| **计算冗余** | $P$ 卡算同一 token 的不同切片 | 无冗余，各卡算不同 token 的不同 expert |
| **内存** | 总参数 $/ P$ | 总参数 $/ P$ |

**直觉解释**：

- **TP 像切蛋糕**：把每个 expert（蛋糕）切成 $P$ 片，每人分到 1 片。要完整结果，$P$ 人都得品尝同一个 token（蛋糕），各报告自己那片的味道，最后汇总。
- **EP 像分蛋糕**：把 $E$ 个 expert（$E$ 块蛋糕）分给 $P$ 人，每人 $E/P$ 块完整的。每个 token（顾客）只去自己喜欢的蛋糕所在的人那里取，不浪费。

##### 14.5 两种拆分的数学等价性

设 $\text{Expert}_e(\mathbf{h})$ 为 expert $e$ 对 token $\mathbf{h}$ 的完整输出。Expert 的 FFN 可分解为三步：

$$\text{Expert}_e(\mathbf{h}) = \underbrace{\mathbf{W}_{\text{down},e}}_{\text{row}} \cdot \underbrace{\left(\text{SiLU}(\mathbf{h} \cdot \mathbf{W}_{\text{gate},e}^\top) \odot (\mathbf{h} \cdot \mathbf{W}_{\text{up},e}^\top)\right)}_{\text{col}}$$

**TP 拆分**：在 column 侧切分 $\mathbf{W}_{\text{gate},e}$ 和 $\mathbf{W}_{\text{up},e}$（每卡 $D_i/P$ 列），再在 row 侧切分 $\mathbf{W}_{\text{down},e}$（每卡 $D_i/P$ 行）。rank $r$ 的 partial 结果为：

$$\text{Expert}_e^{(r)}(\mathbf{h}) = \mathbf{W}_{\text{down},e}^{(r)} \cdot \left(\text{SiLU}(\mathbf{h} \cdot {\mathbf{W}_{\text{gate},e}^{(r)}}^\top) \odot (\mathbf{h} \cdot {\mathbf{W}_{\text{up},e}^{(r)}}^\top)\right) \in \mathbb{R}^{D}$$

由于 SiLU 是非线性激活，**partial 结果不能直接求和得到完整结果**——SiLU 分片后的非线性操作破坏了线性叠加性。TP 之所以可行，是因为 SiLU 作用在每个分片的独立维度上（分片间无交叉），各分片的 $\mathbf{W}_{\text{down}}^{(r)}$ 行也对应各自的分片区域，因此：

$$\text{Expert}_e(\mathbf{h}) = \sum_{r=0}^{P-1} \text{Expert}_e^{(r)}(\mathbf{h})$$

每个 partial 在 intermediate 维度的不同区域独立完成 gate→act→down，在输出维度 $D$ 上直接求和即完整结果。

**EP 拆分**：不切分任何权重矩阵，只按 expert 编号分配。rank $r$ 对 token $\mathbf{h}$ 的贡献为：

$$\mathbf{Y}_r(\mathbf{h}) = \sum_{e \in \text{TopK}(\mathbf{h}) \cap \mathcal{E}_r} g_e \cdot \text{Expert}_e(\mathbf{h})$$

完整结果为所有 rank 贡献之和：

$$\text{MoE}_{\text{routed}}(\mathbf{h}) = \alpha \sum_{r=0}^{P-1} \mathbf{Y}_r(\mathbf{h}) = \alpha \sum_{e \in \text{TopK}(\mathbf{h})} g_e \cdot \text{Expert}_e(\mathbf{h})$$

两种拆分的最终数学结果完全等价，但**通信模式和计算效率不同**。

##### 14.6 并行维度分配公式

对应 [distributed/parallel_state.py:1931](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L1931),[2003](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L2003)。

在 `tp_size` 张卡的单机配置下，attention 和 MoE 的并行维度独立分配：

$$P_{\text{attn\_tp}} = \frac{P_{\text{tp}}}{P_{\text{attn\_cp}} \times P_{\text{attn\_dp}}}$$

$$P_{\text{moe\_tp}} = \frac{P_{\text{tp}}}{P_{\text{moe\_ep}} \times P_{\text{moe\_dp}}}$$

约束：$P_{\text{attn\_tp}} \times P_{\text{attn\_cp}} \times P_{\text{attn\_dp}} = P_{\text{tp}} = P_{\text{moe\_tp}} \times P_{\text{moe\_ep}} \times P_{\text{moe\_dp}}$

**典型配置示例**（$P_{\text{tp}} = 8$）：

| 配置 | $P_{\text{attn\_cp}}$ | $P_{\text{attn\_tp}}$ | $P_{\text{moe\_ep}}$ | $P_{\text{moe\_tp}}$ | 每层通信 |
|------|----------------------|----------------------|---------------------|---------------------|---------|
| 纯 TP | 1 | 8 | 1 | 8 | 2 (TP AllReduce×2) |
| CP + 默认 MoE | 8 | 1 | 1 | 8 | 3 (KV-AG + MLP-AG + RS) |
| CP + DeepEP | 8 | 1 | 8 | 1 | 1 (KV-AG only) |

CP 启用时 $P_{\text{attn\_cp}} = P_{\text{tp}}$，使 $P_{\text{attn\_tp}} = 1$（attention 权重不切分），所有并行度给 CP 切序列。MoE 侧的 $P_{\text{moe\_ep}}$ 独立可调：默认为 1（走 TP 切权重），显式设为 $P_{\text{tp}}$ 后走 EP 切专家。

##### 14.7 EP 组与 TP 组的物理对应

对应 [distributed/parallel_state.py:1888-2070](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L1888-L2070)。

$P_{\text{tp}} = 8$ 时，GPU 编号 0~7。

**TP 组**（[distributed/parallel_state.py:1893](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L1893)）：

$$\text{TP\_group} = [0, 1, 2, 3, 4, 5, 6, 7]$$

所有 8 卡在一个 TP 组内。TP 通信（AllReduce）在这 8 卡间进行。

**EP 组**（当 $P_{\text{moe\_ep}} = 8$, $P_{\text{moe\_tp}} = 1$ 时，[distributed/parallel_state.py:2034](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L2034)）：

$$\text{EP\_group} = [0, 1, 2, 3, 4, 5, 6, 7]$$

当 `moe_ep_size == tensor_model_parallel_size` 时，`_MOE_EP = _TP`（[distributed/parallel_state.py:2034](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L2034)）——EP 组和 TP 组是同一组卡，只是语义不同。

**CP 组**（当 $P_{\text{attn\_cp}} = 8$ 时，[distributed/parallel_state.py:1937](https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/srt/distributed/parallel_state.py#L1937)）：

$$\text{CP\_group} = [0, 1, 2, 3, 4, 5, 6, 7]$$

当 `attn_cp_size == tensor_model_parallel_size` 时，`_ATTN_CP = _TP`——CP 组也是同一组卡。

**结论**：在单机 $P_{\text{tp}} = 8$ 配置下，TP/EP/CP 三组是同一批物理卡的不同逻辑分组，各自的通信操作（AllReduce / a2a / AllGather）都在相同的 NVLink 互连上执行。性能差异来自**通信模式**（dense vs sparse）和**计算效率**（冗余 vs 无冗余），而非物理拓扑。

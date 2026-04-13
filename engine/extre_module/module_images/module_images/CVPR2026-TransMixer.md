# TransMixer 模块总结

---

## 1. 动机

### 现有方法的问题

**CNN 的局限性：**
感受野有限，难以有效建模长距离像素依赖关系，在处理形态复杂的裂缝时性能受限。[1][3]

**Transformer 的局限性：**
虽然能通过注意力机制显式建模长距离依赖，但二次方计算复杂度导致推理效率低下。[1][3]

**Mamba 的局限性：**
具备线性计算复杂度的全局注意力机制，但其渐进式数据处理方式限制了在单次前向传播中捕捉和利用全局上下文的能力。[2]

**现有混合架构的根本缺陷：**
MambaVision、VAMBA、RestorMixer 等混合模型仅将 Mamba 与 Transformer 模块以**串行或并行方式简单堆叠**，未深入分析各架构之间的内在关系与差异，导致多样化架构设计的优势无法被充分发挥。[2][4]

### 提出模块的目的

通过**深入挖掘 Mamba 内部隐式注意力机制**，发现其通道维度上天然存在全局与局部注意力的分化现象，从而设计一个让 CNN、Transformer、Mamba 各司其职、协同互补的混合编码结构，而非简单的模块堆叠。[1][2][6]

---

## 2. 工作原理与核心思想

### 理论基础：Mamba 的隐式注意力

TransMixer 的设计建立在对 Mamba 隐式注意力的深入分析之上。

将 Mamba 的 SSM 递推计算展开，第 $$j$$ 个 token 对第 $$i$$ 个 token 输出的贡献可量化为：

$$y_i = \sum_{j=1}^{i} \alpha_{i,j} \odot x_j$$

其中权重因子：

$$\alpha_{i,j} = C_i^T \prod_{k=j+1}^{i} \bar{A}_k \odot \bar{B}_j$$

这与 Transformer 中的注意力分数在形式上高度类似。[5][6]

进一步分析状态转移方程：

$$h_t = e^{A\Delta_t} h_{t-1} + \Delta_t B_t x_t$$

**关键发现**：$$\Delta_t$$ 决定了历史 token 对当前 token 的影响程度——
- 当 $$\Delta_t \to 0$$ 时，当前 token 被丢弃，保留历史上下文（**局部注意力行为**）
- 当 $$\Delta_t > 0$$ 时，当前 token 与历史信息融合（**全局注意力行为**）

通过对 VMamba 最后一层 Mamba 模块通道维度上注意力权重的可视化观察，发现**全局注意力通道数与局部注意力通道数大致相当**，这为将通道按 $$\gamma = 0.5$$ 的比例解耦提供了理论依据。[6][12]

---

### 模块整体工作流程

TransMixer 的完整计算流程如下：

**Step 1：标准 Mamba 前向计算**

$$Z = \sigma_1(\text{Linear}(I)); \quad X = \text{Conv1D}(\text{Linear}(I))$$

$$Y = \text{SSM}(X)$$

其中 $$Y \in \mathbb{R}^{L \times d}$$ 为 SSM 输出。[4]

**Step 2：基于 $$\Delta_t$$ 的通道解耦**

沿通道维度对 $$\Delta_t$$ 排序，选取前 $$d_g = d \cdot \gamma$$ 个通道作为**全局 token**，其余 $$d_l = d \cdot (1-\gamma)$$ 个通道作为**局部 token**：

$$g, l = f_{index}(\Delta_t)$$

$$Y_G, Y_L = Y[:, g], \quad Y[:, l]$$

[6][7]

**Step 3a：全局 token 处理（Transformer 路径）**

将全局表示 $$Y_G \in \mathbb{R}^{L \times d_g}$$ 送入 **Self-Attention 模块**，进一步建模全局 token 之间的长距离依赖关系，得到增强表示 $$Y'_G$$。[7]

**Step 3b：局部 token 处理（CNN 路径）**

将局部表示 $$Y_L \in \mathbb{R}^{L \times d_l}$$ 送入轻量级 **Local Refinement Module**：

$$F_L \in \mathbb{R}^{d_l \times H \times W} = \text{Norm}(\text{Reshape}(Y_L))$$

$$F'_L = \sigma_2(\text{Conv}_{1\times1}(\text{maxpool2d}(F_L))) \odot F_L$$

$$Y'_l = \text{Flatten}(F'_L)$$

通过 MaxPooling 提取局部区域内最显著的裂缝特征，再通过逐元素乘法实现局部细粒度特征增强。[7]

**Step 4：合并输出**

将增强后的全局与局部表示写回对应通道位置，执行最终线性投影：

$$Y[:, g], Y[:, l] = f_{\text{self-att}}(Y_G), \quad f_{\text{refine}}(Y_L)$$

$$O = \text{Linear}(Y \odot Z)$$

最终得到特征图 $$F_i \in \mathbb{R}^{C_i \times H_i \times W_i}$$。[7]

---

### 三条路径的角色分工

| 路径 | 架构来源 | 处理对象 | 核心职责 |
|------|---------|---------|---------|
| **Mamba 路径** | SSM | 全部 token | 序列上下文建模，生成全局/局部分化的初始表示 |
| **Transformer 路径** | Self-Attention | 全局 token | 强化全局 token 间的长距离依赖关系 |
| **CNN 路径** | MaxPool + Conv | 局部 token | 增强局部细粒度纹理特征 |

[6][7]

---

## 3. 总结

| 维度 | 内容 |
|------|------|
| **核心创新** | 不再简单堆叠模块，而是**基于 Mamba 隐式注意力的内在分化**，让全局与局部通道自然对应 Transformer 和 CNN 路径，实现有理论依据的协同设计 [2][6] |
| **设计哲学** | 将混合架构从"堆叠的混合体"升级为"专家协作团队"——每种架构在最适合自己的位置发挥作用 [1] |
| **超参数合理性** | $$\gamma = 0.5$$ 由 Mamba 通道注意力可视化观察得出，实验验证偏离 0.5 均导致性能下降 [12][13] |
| **消融实验验证** | 移除 TransMixer 后 mIoU 在 DeepCrack 下降 0.76%；单独移除全局或局部 token 均导致不同程度的性能下滑，证明两条路径缺一不可 [19] |
| **对比竞品** | 相比使用 MambaVision 和 RestorMixer 作为编码器，TransMixer 在 DeepCrack 和 CamCrack789 上均取得最优分割性能 [13] |
| **Local Refinement 细节** | MaxPooling 比 AvgPooling 更有效，因为裂缝特征稀疏且显著，MaxPooling 能更好地捕捉局部区域内最关键的裂缝响应 [19] |
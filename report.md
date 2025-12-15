# JC4001 Distributed Systems - Federated Learning Assignment Report

**Student Name:** zishanxu
**Student ID:** 50087477
**Date:** December 2025

---

## Part 1: Understanding Federated Learning

### Summary of Federated Learning Principles

McMahan et al. (2017) introduced Federated Learning as a practical solution to a fundamental challenge in modern machine learning: leveraging the wealth of data on mobile devices while preserving user privacy. Modern smartphones and tablets contain unprecedented amounts of valuable training data—from photos and text messages to speech patterns and behavioral information—yet this data is often too privacy-sensitive or voluminous to be logged to centralized data centers using conventional approaches.

Federated Learning enables users to collectively benefit from shared models trained on rich distributed data without centralizing storage. The approach coordinates a loose federation of participating devices (clients) through a central server. Each client maintains its local training dataset which is never uploaded; instead, clients compute updates to the current global model and communicate only these updates to the server. This directly applies the principle of focused collection or data minimization, as these specific updates need not be stored once applied to improve the model.

The **FederatedAveraging (FedAvg) algorithm** forms the core contribution of this work. It combines local stochastic gradient descent (SGD) on each client with server-side model averaging. In each communication round, the server selects a random fraction C of clients and broadcasts the current global model. Selected clients then perform E local training epochs on their data using minibatch size B, and return their updated model weights to the server. The server aggregates these weights using weighted averaging based on the number of samples each client holds: w_global = Σ(n_k/n_total) × w_k, where n_k represents client k's dataset size. This weighted aggregation ensures clients with more data have proportionally greater influence.

**Federated optimization** differs fundamentally from traditional distributed optimization. Key distinguishing properties include: (1) Non-IID data—each client's data reflects individual usage patterns rather than representing the population distribution; (2) Unbalanced data—some users generate far more data than others; (3) Massively distributed—the number of participating clients vastly exceeds the average examples per client; and (4) Limited communication—mobile devices are frequently offline or on slow, expensive connections. Communication costs dominate computational costs in this setting, making communication efficiency the principal constraint.

The paper's extensive empirical evaluation across five model architectures (including multi-layer perceptrons, CNNs, and LSTMs) and four datasets demonstrates FedAvg's remarkable communication efficiency. The algorithm reduces required communication rounds by 10-100× compared to baseline federated SGD, achieving this through increased local computation per client. Experiments show the approach is robust to unbalanced and non-IID data distributions—though non-IID data remains the most challenging scenario, causing slower convergence and reduced accuracy.

**Privacy advantages** distinguish federated learning from centralized approaches. Raw data never leaves devices, eliminating risks from holding "anonymized" datasets that remain vulnerable to privacy attacks through data joins. Updates transmitted contain only the minimal information needed for model improvement and can be sent anonymously over mix networks. The work suggests future integration with differential privacy and secure multiparty computation could provide even stronger guarantees. The success of FedAvg on real-world problems, including a large-scale next-word prediction task with over 500,000 clients, demonstrates federated learning's practical viability for privacy-preserving distributed machine learning.

---

## Part 2: Centralised Learning Baseline

### Implementation Details

The centralized learning baseline was implemented using a fully connected neural network architecture with the Fashion-MNIST dataset. This serves as a benchmark for comparing federated learning performance.

**Dataset:** Fashion-MNIST contains 60,000 training images and 10,000 test images of 10 clothing categories (28x28 grayscale images).

**Model Architecture:**

The neural network model follows a fully connected architecture:

**Algorithm: FashionMNIST Neural Network**
```
Input: x ∈ ℝ^(28×28) (grayscale image)
Output: ŷ ∈ ℝ^10 (class probabilities)

1: x ← Flatten(x)                    // x ∈ ℝ^784
2: h₁ ← ReLU(W₁x + b₁)              // W₁ ∈ ℝ^(128×784)
3: h₂ ← ReLU(W₂h₁ + b₂)             // W₂ ∈ ℝ^(64×128)
4: ŷ ← W₃h₂ + b₃                     // W₃ ∈ ℝ^(10×64)
5: return ŷ
```

Network layers:
- Input layer: 784 neurons (flattened 28×28 images)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 10 neurons (class logits)

**训练配置与超参数设置**

本实验采用 PyTorch 深度学习框架实现集中式学习基线。模型使用 Adam 优化器进行参数更新，学习率设定为 0.001，以平衡收敛速度与稳定性。损失函数选择交叉熵损失（CrossEntropyLoss），适用于多分类任务。为充分利用硬件并行计算能力，批次大小设置为 1024。训练过程总计进行 15 个 epoch，确保模型充分收敛。表 1 总结了完整的训练超参数配置。

**Table 1: Centralised Learning Hyperparameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Framework | PyTorch | Deep learning framework |
| Optimizer | Adam | Adaptive moment estimation |
| Learning Rate (η) | 0.001 | Step size for gradient descent |
| Loss Function | CrossEntropyLoss | Multi-class classification loss |
| Batch Size (B) | 1024 | Samples per mini-batch |
| Epochs | 15 | Complete passes through dataset |
| Device | CPU/CUDA | Auto-detected based on availability |

**实验结果分析**

最近一次完整运行的实验结果表明，集中式学习模型在训练集上达到 88.33% 的准确率，在测试集上获得 86.76% 的准确率，测试损失为 0.3692。整个训练过程耗时 25.70 秒，展现出良好的计算效率。观察训练曲线可发现，模型在前 10 个 epoch 内快速收敛，准确率从初始的约 70% 提升至 85% 以上，随后在剩余的 epoch 中继续缓慢提升并趋于稳定。这一收敛特性为后续联邦学习实验提供了重要的性能基准参考。

---

## Part 3: Federated Learning Implementation

### A. Client Simulation and Data Partitioning

联邦学习系统的数据分布策略直接影响模型的收敛性能和最终准确率。本实验实现了两种数据分区方案，分别模拟理想化和现实化的分布式场景。

**IID Data Partitioning (独立同分布)**

在 IID 场景下，60,000 个训练样本首先被随机打乱，然后均匀分配给 K 个客户端，每个客户端获得 n = 60000/K 个样本。通过分层采样（stratified sampling）策略，确保每个客户端的本地数据集在类别分布上近似于全局数据分布。具体而言，对每个类别 c ∈ {0,1,...,9}，将该类别的所有样本随机分配给所有客户端，使得每个客户端获得的该类别样本数大致相等。这种分区方式模拟了理想化的协作学习场景，其中所有参与方的数据特征统计上相似。

**Non-IID Data Partitioning (非独立同分布)**

为模拟真实世界中数据异构性（heterogeneity）带来的挑战，本实验采用基于类别偏斜（class skew）的分区策略。每个客户端仅被分配来自 C 个类别的数据（默认 C=2），而非全部 10 个类别。这种设计创建了统计异质性：不同客户端"专注"于不同的服装类别，例如客户端 1 可能仅持有"T-shirt"和"Trouser"的图像，而客户端 2 持有"Dress"和"Sandal"。为确保实验的完整性，算法保证所有 10 个类别至少被分配给一个客户端，避免全局模型在某些类别上缺乏训练数据。表 2 对比了两种分区策略的关键特性。

**Table 2: Comparison of Data Partitioning Strategies**

| Characteristic | IID Partitioning | Non-IID Partitioning |
|----------------|------------------|----------------------|
| Class Distribution | Uniform across clients | Skewed (2-3 classes per client) |
| Statistical Similarity | High (clients similar to global) | Low (heterogeneous) |
| Real-world Accuracy | Low (idealized scenario) | High (realistic scenario) |
| Convergence Behavior | Fast, stable | Slow, oscillating |
| Expected Performance | ~82-86% accuracy | ~36-57% accuracy |
| Implementation Complexity | Simple random split | Class-based assignment |

### B. Local Model Training

在联邦学习框架中，每个客户端在本地数据上独立执行模型训练，无需将原始数据上传至中央服务器。训练开始时，客户端 k 从服务器接收当前全局模型参数 w_global 的副本，并使用深拷贝（deep copy）创建本地模型实例以避免参数污染。随后，客户端在其私有数据集 D_k 上执行 E 个 epoch 的随机梯度下降（SGD）训练。

在每个 epoch 中，本地数据被分成若干小批次（mini-batch），批次大小为 B。对于每个批次 (x, y)，模型执行前向传播计算预测输出 ŷ，然后计算交叉熵损失 L(ŷ, y)。反向传播算法计算损失关于模型参数的梯度 ∇_w L，最后通过 SGD 更新规则 w_k ← w_k - η·g 更新参数，其中 η 为学习率。经过 E 个 epoch 的本地训练后，客户端将更新后的模型参数 w_k 返回给服务器进行聚合。整个过程中，原始训练数据 D_k 始终保留在客户端本地，实现了数据隐私保护。

**Algorithm: Client Local Training**
```
Input: w_global (global model weights), D_k (local dataset), E (local epochs)
Output: w_k (updated local model weights)

1: w_k ← w_global                        // Initialize local model
2: for epoch = 1 to E do
3:     for each batch (x, y) ∈ D_k do
4:         ŷ ← Forward(w_k, x)           // Forward pass
5:         L ← CrossEntropy(ŷ, y)        // Compute loss
6:         g ← ∇_w L                      // Backward pass (compute gradients)
7:         w_k ← w_k - η·g               // SGD update (η = learning rate)
8:     end for
9: end for
10: return w_k
```

### C. FedAvg Server-Side Aggregation

联邦平均（Federated Averaging, FedAvg）算法是联邦学习的核心聚合机制，负责将来自多个客户端的本地模型更新融合为统一的全局模型。当服务器从所有参与客户端收集到更新后的模型参数 {w_1, w_2, ..., w_K} 以及对应的数据集大小 {n_1, n_2, ..., n_K} 后，执行加权平均操作。

聚合过程首先计算所有客户端的总样本数 n_total = Σ_{k=1}^K n_k，然后为每个客户端 k 计算权重系数 p_k = n_k / n_total，该系数表示客户端 k 的数据量在全局数据中的占比。全局模型参数通过加权累加得到：w_global = Σ_{k=1}^K p_k · w_k。这种加权策略确保了持有更多训练数据的客户端对全局模型有更大的影响力，符合统计学习理论中样本量与估计精度的关系。算法 3 详细描述了 FedAvg 的计算流程。

**Algorithm: Federated Averaging (FedAvg)**
```
Input: {w_k}_{k=1}^K (client model weights), {n_k}_{k=1}^K (client data sizes)
Output: w_global (aggregated global model weights)

1: n_total ← Σ_{k=1}^K n_k                // Total number of samples
2: w_global ← 0
3: for k = 1 to K do
4:     p_k ← n_k / n_total                 // Compute weight coefficient
5:     w_global ← w_global + p_k · w_k     // Weighted accumulation
6: end for
7: return w_global
```

**Table 3: FedAvg Weighting Examples**

| Client | Local Samples (n_k) | Weight (p_k) | Influence |
|--------|---------------------|--------------|-----------|
| Client 1 | 10,000 | 0.167 | 16.7% |
| Client 2 | 15,000 | 0.250 | 25.0% |
| Client 3 | 20,000 | 0.333 | 33.3% |
| Client 4 | 15,000 | 0.250 | 25.0% |
| **Total** | **60,000** | **1.000** | **100%** |

表 3 展示了一个 4 客户端场景下的权重分配示例。拥有最多数据的 Client 3 (20,000 samples) 在聚合中占据 33.3% 的权重，而 Client 1 仅占 16.7%。

### D. Communication Rounds and System Workflow

联邦学习训练过程通过多轮（rounds）客户端-服务器通信迭代完成。每一轮通信包含四个关键阶段：模型广播、本地训练、参数上传和全局聚合。本实验配置总计执行 T = 20 轮通信，每轮中所有客户端执行 E = 3 个本地 epoch 的训练。

在每轮通信 t 开始时，服务器向所有 K 个客户端广播当前全局模型参数 w_t。各客户端并行地在本地数据集上训练模型，执行 E 次完整的数据遍历（epoch）。训练完成后，客户端将更新后的本地模型参数 w_k^{(t)} 上传至服务器。服务器收集所有客户端的参数更新后，调用 FedAvg 算法计算加权平均，得到新的全局模型 w_{t+1} = Σ_{k=1}^K (n_k/n_total) · w_k^{(t)}。这一全局模型随后在测试集上进行评估，记录准确率和损失值。图 1 展示了单轮通信的完整数据流。

**Figure 1: Federated Learning Communication Flow (Single Round)**

```
┌─────────────────────────────────────────────────────────────────┐
│                         Server (Round t)                         │
│  ┌────────────────┐         ┌────────────────┐                  │
│  │ Global Model   │  ─────> │   Broadcast    │                  │
│  │  w_t           │         │   w_t to all   │                  │
│  └────────────────┘         └────────────────┘                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ (1) Model Distribution
                 ├────────────────┬────────────────┬──────────────┐
                 ▼                ▼                ▼              ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     ...
      │  Client 1    │  │  Client 2    │  │  Client K    │
      │  D_1, n_1    │  │  D_2, n_2    │  │  D_K, n_K    │
      ├──────────────┤  ├──────────────┤  ├──────────────┤
      │ w_1 ← w_t    │  │ w_2 ← w_t    │  │ w_K ← w_t    │
      │              │  │              │  │              │
      │ (2) Local    │  │ (2) Local    │  │ (2) Local    │
      │ Training:    │  │ Training:    │  │ Training:    │
      │ E epochs     │  │ E epochs     │  │ E epochs     │
      │ on D_1       │  │ on D_2       │  │ on D_K       │
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             │                 │                 │
             │ (3) Upload      │ Upload          │ Upload
             │ w_1^(t)         │ w_2^(t)         │ w_K^(t)
             └─────────────────┴─────────────────┴──────────┐
                                                             │
┌────────────────────────────────────────────────────────────┴────┐
│                       Server Aggregation                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ (4) FedAvg: w_{t+1} = Σ (n_k/n_total) · w_k^(t)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ (5) Evaluate w_{t+1} on test set                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Table 4: Communication Round Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Total Rounds (T) | 20 | Number of communication iterations |
| Local Epochs (E) | 3 | Training epochs per client per round |
| Client Participation | 100% | Fraction of clients selected each round |
| Total Local Updates | 60 | T × E = 20 × 3 |
| Communication Cost | 20 rounds | Model transmissions (up + down) |

表 4 总结了通信轮次的关键参数。相比集中式学习的 15 epochs，联邦学习通过 20 轮 × 3 本地 epochs 共执行 60 次本地数据遍历，但仅需 20 次模型通信。

---

## Part 4: Experimentation and Analysis

### Experiment 1: Impact of Client Count (IID Data Distribution)

本实验探究客户端数量对联邦学习性能的影响。在 IID 数据分布条件下，分别配置 5、10 和 20 个客户端参与训练。所有实验采用统一的超参数设置：每轮通信所有客户端 100% 参与，每个客户端执行 3 个本地 epoch，总计 20 轮通信。数据分区采用分层采样策略，确保每个客户端的类别分布与全局分布一致。

表 5 总结了不同客户端数量下的实验结果。可以观察到，随着客户端数量从 5 增加到 20，测试准确率显著下降（86.43% → 82.05%），准确率损失达 4.38 个百分点。这一现象可归因于两个因素：（1）客户端数量增加导致每个客户端的本地数据量减少（从 12,000 降至 3,000 样本），本地模型的训练质量下降；（2）更多客户端的聚合过程引入更多的模型方差。训练时间随客户端数量线性增长，从 5 客户端的 173 秒增至 20 客户端的 590 秒，这是由于模拟环境中客户端串行训练所致（真实分布式环境可并行化）。

**Table 5: Performance vs. Number of Clients (IID)**

| Clients (K) | Samples/Client | Test Accuracy | Test Loss | Training Time | Speedup vs. Centralized |
|-------------|----------------|---------------|-----------|---------------|-------------------------|
| 5           | 12,000         | 86.43%        | 0.3821    | 173.24 s      | 0.15× (slower) |
| 10          | 6,000          | 85.01%        | 0.4241    | 301.52 s      | 0.09× (slower) |
| 20          | 3,000          | 82.05%        | 0.5133    | 589.84 s      | 0.04× (slower) |
| **Centralized** | **60,000** | **86.76%** | **0.3692** | **25.70 s** | **1.0× (baseline)** |

### Experiment 2: Impact of Non-IID Data Distribution

本实验对比 IID 与 Non-IID 数据分布对联邦学习性能的影响。Non-IID 场景配置 10 个客户端，每个客户端仅持有 C = 2 个类别的数据，模拟真实世界中数据异构性。例如，Client 1 可能仅有"Ankle boot"和"Bag"类别，而 Client 2 持有"Dress"和"Pullover"。其余训练超参数与 IID 实验保持一致。

实验结果揭示了 Non-IID 数据对联邦学习的灾难性负面影响。如表 6 所示，Non-IID 场景下的测试准确率仅为 41.24%，相比 IID 的 85.01% 下降了 43.77 个百分点（相对下降 51.5%）。这一显著性能退化源于客户端漂移（client drift）现象：每个客户端在本地偏斜数据上训练时，模型参数朝着局部最优方向更新，而这些方向在不同客户端之间可能相互冲突。聚合后的全局模型难以在所有类别上达到良好泛化性能。训练曲线显示 Non-IID 场景下准确率震荡剧烈，收敛不稳定。

**Table 6: IID vs. Non-IID Performance (10 Clients)**

| Metric | IID | Non-IID | Degradation |
|--------|-----|---------|-------------|
| Test Accuracy | 85.01% | 41.24% | **-43.77 pp** (↓51.5%) |
| Test Loss | 0.4241 | 2.9292 | +2.5051 (↑590%) |
| Training Time | 301.52 s | 307.08 s | +5.56 s (↑1.8%) |
| Convergence | Stable | Oscillating | Unstable |
| Classes per Client | 10 (all) | 2 (limited) | -8 classes |

表 6 清晰展示了数据分布异构性对联邦学习的毁灭性影响，准确率下降达 44 个百分点，损失增加近 6 倍。

---

## Part 5: Performance Comparison with Centralised Learning

### Comprehensive Performance Analysis

本节对集中式学习与多种联邦学习配置进行全面性能对比分析。实验涵盖了不同客户端数量（5/10/20）和数据分布类型（IID/Non-IID）的组合，共 4 个联邦学习场景。表 7 汇总了所有实验配置的关键性能指标。

**Table 7: Comprehensive Performance Comparison**

| Method | Clients | Data Dist. | Test Acc. | Test Loss | Time (s) | Comm. Rounds | Privacy |
|--------|---------|------------|-----------|-----------|----------|--------------|---------|
| **Centralized** | - | Centralized | 86.76% | 0.3692 | 25.70 | - | ✗ No |
| FL | 5 | IID | 86.43% | 0.3821 | 173.24 | 20 | ✓ Yes |
| FL | 10 | IID | 85.01% | 0.4241 | 301.52 | 20 | ✓ Yes |
| FL | 20 | IID | 82.05% | 0.5133 | 589.84 | 20 | ✓ Yes |
| FL | 5 | **Non-IID** | 57.32% | 1.1860 | 173.55 | 20 | ✓ Yes |
| FL | 10 | **Non-IID** | **41.24%** | **2.9292** | 307.08 | 20 | ✓ Yes |
| FL | 20 | **Non-IID** | **35.86%** | **2.2074** | 583.60 | 20 | ✓ Yes |

### Analysis and Key Findings

**准确率对比分析**

实验结果表明，在 IID 数据分布下，联邦学习的准确率与集中式学习基本持平，仅有轻微下降（86.76% vs. 85.01% for 10 clients）。5 客户端场景达到 86.43%，与集中式学习仅相差 0.33 个百分点。这一结果证明了 FedAvg 算法在理想数据分布条件下的有效性。然而，Non-IID 场景呈现截然不同的结果：10 客户端配置下准确率暴跌至 41.24%，相比集中式学习下降 45.52 个百分点；20 客户端配置更是降至 35.86%，仅为集中式学习的 41%。这一"灾难性失败"充分展示了数据异构性对联邦学习的破坏性影响，也解释了为何 Non-IID 问题是当前联邦学习研究的核心挑战之一。

**训练效率与时间成本**

从计算时间角度看，集中式学习展现出显著优势，仅需 25.70 秒完成训练，而联邦学习场景耗时 173-590 秒，慢 6.7-23.0 倍。这一开销主要源于两个因素：（1）多轮通信带来的迭代次数增加（20 轮 vs. 15 epochs）；（2）模拟环境中客户端串行训练的限制。需要强调的是，在真实分布式部署中，各客户端可并行训练，墙钟时间（wall-clock time）将大幅缩短。因此，表中的时间对比更多反映了总计算量而非实际延迟。Non-IID 场景的训练时间（307.08s）与 IID（301.52s）相近，说明数据分布主要影响收敛质量而非计算速度。

**收敛特性观察**

集中式学习表现出平滑的单调收敛曲线，在前 10 个 epoch 内快速提升至 85% 以上准确率，随后缓慢趋于稳定。IID 联邦学习的收敛过程相似但速度稍慢，大部分性能提升发生在前 10 轮通信中。相比之下，Non-IID 场景展现出严重的震荡不稳定性：测试准确率在连续轮次间可能上下波动 5-10 个百分点，反映了不同客户端梯度方向的剧烈冲突。这种不稳定性使得早停（early stopping）等优化策略难以应用。

**隐私-性能权衡分析**

联邦学习的核心价值在于在保护数据隐私的前提下实现机器学习。IID 场景下的实验结果令人鼓舞：以仅 0.33-4.71 个百分点的准确率损失（相对下降 0.4-5.4%）为代价，获得了完整的数据隐私保护。5 客户端配置几乎达到集中式学习的性能（86.43% vs. 86.76%），这表明在数据分布理想的情况下，隐私与性能并非不可调和的矛盾。然而，Non-IID 场景揭示了严峻的现实：隐私保护的代价高达 45-51 个百分点的准确率损失，使得模型几乎无法实用。10 客户端 Non-IID 配置仅达到 41.24% 准确率，甚至低于随机猜测的 10% 仅 31 个百分点。解决 Non-IID 挑战需要采用高级技术，包括：（1）增加通信轮次以克服客户端漂移；（2）使用自适应学习率调度；（3）客户端聚类（clustering）或个性化（personalization）策略；（4）有限的数据共享方案（如共享小型公共数据集）。

**实践意义与启示**

本实验为联邦学习的实际部署提供了重要洞察。首先，IID 数据场景证明了联邦学习的可行性，为医疗、金融等隐私敏感领域的应用铺平了道路。其次，Non-IID 问题的严重性警示从业者必须在部署前评估数据异构性，并采取针对性缓解措施。第三，通信轮次的设计需要权衡收敛质量与通信开销，本实验的 20 轮配置在 IID 场景下已基本充分。最后，客户端数量的增加显著影响性能：IID 场景下从 5 客户端的 86.43% 降至 20 客户端的 82.05%，准确率损失 4.38 个百分点；Non-IID 场景更严重，从 5 客户端的 57.32% 暴跌至 20 客户端的 35.86%，损失超过 21 个百分点。实际部署时应根据隐私需求、数据分布和性能要求选择合适的客户端规模。

---

## 分布式系统中联邦学习的整体设计 (Overall Design of Federated Learning in Distributed Systems)

### 系统架构 (System Architecture)

本实现采用**客户端-服务器（Client-Server）分布式架构**，模拟真实的联邦学习场景。系统由三个核心组件构成：

#### 1. 服务器端 (Server-Side Components)

**FederatedServer 类**负责全局模型管理和协调：

- **全局模型维护**：存储和管理当前的全局模型参数 `w_global`
- **客户端选择**：每轮随机选择参与训练的客户端子集（client fraction C）
- **模型分发**：向选中的客户端广播当前全局模型
- **聚合算法**：实现 FedAvg 加权平均聚合
  ```
  w_global(t+1) = Σ(n_k / n_total) × w_k(t+1)
  ```
  其中 `n_k` 是客户端 k 的样本数，`w_k` 是其本地模型参数
- **模型评估**：在全局测试集上评估聚合后的模型性能

#### 2. 客户端 (Client-Side Components)

**FederatedClient 类**模拟分布式环境中的独立设备：

- **本地数据存储**：每个客户端维护私有数据集（永不上传）
  - 数据集 D_k = {(x_i, y_i)}_{i=1}^{n_k}
  - 使用数据加载器进行批次采样

- **模型克隆**：接收服务器广播的全局模型并创建本地副本
  ```
  w_k ← Clone(w_global)
  optimizer ← Adam(w_k, lr=η)
  ```

- **本地训练**：
  - 在本地数据上执行 E 个 epoch 的 SGD 训练
  - 使用小批量（batch size B）进行参数更新
  - 独立优化本地目标函数 F_k(w)
  ```
  for epoch = 1 to E:
      for batch (x, y) ∈ D_k:
          g ← ∇_w Loss(Forward(w_k, x), y)
          w_k ← w_k - η·g
  ```

- **参数上传**：仅向服务器发送训练后的模型参数（非原始数据）
  ```
  return w_k
  ```

#### 3. 数据分区模块 (Data Partitioning Module)

实现两种数据分布策略以模拟不同的分布式场景：

**IID 分区（Independent and Identically Distributed）：**
- 随机打乱全部训练数据
- 均匀分配给 K 个客户端
- 每个客户端的数据分布近似总体分布
- 模拟理想化的同质数据场景

**Non-IID 分区（Non-Independent and Identically Distributed）：**
- 按标签对数据排序分组
- 每个客户端仅分配 2-3 个类别的数据
- 模拟真实世界的数据异构性（如不同医院、不同用户习惯）
- 创建统计异质性挑战

### 分布式训练流程 (Distributed Training Workflow)

系统采用**同步联邦学习**范式，每个通信轮次包含以下步骤：

```
第 t 轮通信：
1. [服务器] 选择客户端子集 S_t（大小为 C × K）
2. [服务器→客户端] 广播全局模型 w_t 到选中客户端
3. [客户端] 并行执行本地训练：
   For each client k ∈ S_t:
     w_k ← w_t
     For epoch i = 1 to E:
       For batch b ∈ local_data:
         w_k ← w_k - η∇F_k(w_k; b)
4. [客户端→服务器] 上传本地模型参数 {w_k | k ∈ S_t}
5. [服务器] FedAvg 聚合：
   w_(t+1) ← Σ(n_k/n_total) × w_k
6. [服务器] 在测试集上评估 w_(t+1)
7. 重复直至收敛或达到最大轮次
```

### 关键设计决策 (Key Design Decisions)

#### 通信效率优化 (Communication Efficiency)

- **批量本地更新**：每轮执行多个 epoch（E > 1）减少通信频率
- **部分客户端参与**：C < 1.0 降低每轮通信开销
- **参数级通信**：仅传输模型参数，不传输梯度历史

#### 系统实现方式 (Implementation Approach)

- **模拟 vs. 真实分布式**：采用单进程模拟而非真实网络通信（gRPC/sockets）
  - 优点：可重复性强，易于调试和实验
  - 缺点：无法反映真实网络延迟和故障
- **同步更新策略**：所有选中客户端完成训练后才聚合
  - 简化实现和分析
  - 避免异步更新的陈旧梯度问题

#### 隐私保护机制 (Privacy Protection)

- **数据本地化**：原始数据永不离开客户端设备
- **最小化信息传输**：仅上传模型参数更新
- **无客户端身份关联**：聚合过程不需要客户端元数据

---

## 实现过程中遇到的挑战及解决方法 (Challenges and Solutions)

### 挑战 1: Non-IID 数据分区的实现 (Non-IID Data Partitioning)

**问题描述：**
- 如何创建真实反映分布式异构性的非IID数据分布？
- 如何确保每个客户端有足够数据进行有效训练？
- 如何避免某些类别完全缺失或数据分配不均？

**解决方案：**

**Algorithm: Non-IID Data Partitioning**
```
Input: Dataset D, num_clients K, classes_per_client C
Output: {D_1, D_2, ..., D_K} (client data subsets)

1: // Group data by class
2: for c = 0 to 9 do
3:     I_c ← {i | y_i = c}              // Indices of samples with label c
4: end for
5:
6: // Assign classes to clients
7: A ← [∅, ∅, ..., ∅]                   // A[k] = set of classes for client k
8: classes ← RandomPermutation([0,1,...,9])
9: for each class c in classes do
10:    available ← {k | |A[k]| < C}     // Clients with room for more classes
11:    k ← RandomChoice(available)
12:    A[k] ← A[k] ∪ {c}                // Assign class c to client k
13: end for
14:
15: // Build client datasets
16: for k = 1 to K do
17:    D_k ← ∅
18:    for each c ∈ A[k] do
19:        D_k ← D_k ∪ {(x_i, y_i) | i ∈ I_c}
20:    end for
21:    Shuffle(D_k)
22: end for
23: return {D_1, D_2, ..., D_K}
```

**实现细节：**
- 每个客户端分配 C 个类别（默认 C=2）
- 保证所有10个类别至少分配给一个客户端
- 最终每个客户端数据量可能不平衡（反映真实场景）

**验证效果：**
- Non-IID 场景下准确率从 85.01% (IID) 降至 41.24% (Non-IID)（预期行为）
- 证明成功创建了具有挑战性的数据异构性

### 挑战 2: FedAvg 聚合算法的正确实现 (Correct FedAvg Aggregation)

**问题描述：**
- 神经网络有多层，每层有不同形状的参数（权重矩阵、偏置向量）
- 如何正确提取、加权平均、然后重新设置这些参数？
- 如何确保加权平均考虑客户端数据量差异？

**解决方案：**

FedAvg 算法的数学原理已在 Part 3.C 中描述（见 Algorithm: Federated Averaging）。实现时需要注意以下细节：

**实现关键点：**

1. **参数提取**：
   - 从模型中提取所有层的参数（权重和偏置）
   - 保持参数的原始形状和顺序
   - 每层参数作为独立的张量处理

2. **逐层聚合**：
   - 对每一层 l ∈ {1, 2, ..., L} 分别执行加权平均
   - 权重系数：p_k = n_k / n_total（客户端 k 的样本数占比）
   - 聚合公式：w_global[l] = Σ_{k=1}^K p_k · w_k[l]

3. **参数形状验证**：
   - 确保所有客户端的对应层参数形状一致
   - 第1层：W₁ ∈ ℝ^(128×784), b₁ ∈ ℝ^128
   - 第2层：W₂ ∈ ℝ^(64×128), b₂ ∈ ℝ^64
   - 第3层：W₃ ∈ ℝ^(10×64), b₃ ∈ ℝ^10

4. **数值稳定性**：
   - 使用浮点数累加时注意精度问题
   - 验证权重系数之和：Σ p_k = 1.0

**测试验证：**
- 手动验证小规模场景（2客户端）下的聚合结果
- 确认聚合后模型在测试集上的性能合理
- 验证边界情况：单客户端（应返回原参数）、均等数据量（应为简单平均）

### 挑战 3: 收敛性问题与训练不稳定 (Convergence Issues)

**问题描述：**
- Non-IID 场景下训练曲线剧烈震荡
- 某些配置下准确率不升反降
- 大 E 值（过多本地 epoch）导致发散

**根本原因分析：**
- **客户端漂移（Client Drift）**：本地模型过度拟合本地数据，偏离全局最优
- **梯度冲突**：不同客户端的梯度方向相反（因数据分布不同）
- **学习率不匹配**：固定学习率在聚合后可能过大

**解决方案：**

1. **限制本地 epoch 数**：
   - 本实验采用 E = 3，在 IID 和 Non-IID 场景下均表现稳定
   - 经验表明：IID 数据可使用 E = 3-5；Non-IID 数据建议 E = 1-3
   - 过大的 E 导致客户端漂移加剧，尤其在 Non-IID 场景下

2. **学习率调优**：
   - 使用网格搜索找到最佳学习率
   - 对不同 E 值使用不同学习率
   - 考虑学习率衰减策略

3. **增加通信轮次**：
   - Non-IID 需要更多轮次收敛（20-40 轮 vs. IID 的 10-20 轮）
   - 更频繁的聚合减少局部偏差累积

**实验结果：**
- IID: 20 轮收敛至 82-86% 准确率（取决于客户端数量）
- Non-IID: 20 轮仅达到 36-57% 准确率（需要更多轮次或改进算法）

### 挑战 4: 计算资源与训练时间 (Computational Resources)

**问题描述：**
- 模拟 20 客户端 × 20 轮 × 3 epoch = 1200 次模型训练
- 单次完整实验耗时 15-30 分钟
- 超参数搜索需要运行数百次实验

**优化策略：**

1. **GPU/CPU 选择**：
   - 支持命令行参数 `--cpu` 强制使用 CPU 模式
   - 自动检测 GPU 可用性并选择合适的计算设备
   - 小模型可在 CPU 上快速训练
   - 大规模实验可利用 GPU 加速

2. **减少日志输出**：
   - 训练时设置 `verbose=0` 减少控制台输出
   - 仅在关键节点记录性能指标

3. **代码优化**：
   - 使用多进程数据加载 (`num_workers=8`) 并行读取数据
   - 合理设置 batch size（1024）利用硬件并行性
   - 使用向量化操作避免显式循环

4. **实验设计**：
   - 先在小规模（5 客户端，10 轮）上测试
   - 确认可行后再运行完整实验

**最终性能：**
- 单次完整实验（所有配置）：约 40-60 分钟
- 可接受的时间成本用于学术实验

### 挑战 5: 可重复性保证 (Reproducibility)

**问题描述：**
- 随机数据分区导致每次结果不同
- 模型初始化的随机性
- PyTorch 内部随机性（包括 CUDA）

**解决方案：**

设置所有随机数生成器的种子以确保实验可重复性：

**Random Seed Initialization:**
```
1: RANDOM_SEED ← 42                      // Unified seed for all experiments
2:
3: // Set seeds for all random number generators
4: random.seed(RANDOM_SEED)              // Python random module
5: np.random.seed(RANDOM_SEED)           // NumPy random
6: torch.manual_seed(RANDOM_SEED)        // PyTorch CPU
7:
8: // Additional seeds for GPU if available
9: if GPU_Available() then
10:    torch.cuda.manual_seed(RANDOM_SEED)
11:    torch.cuda.manual_seed_all(RANDOM_SEED)
12:    // Enable deterministic behavior
13:    cudnn.deterministic ← True
14:    cudnn.benchmark ← False
15: end if
```

**实现要点：**
- 统一随机种子 `RANDOM_SEED = 42` 定义在配置文件中
- 在数据分区、模型初始化前设置种子
- GPU 确定性模式可能略微降低性能但保证可重复性

**验证方法：**
- 多次运行相同配置，确认结果一致
- 记录每次实验的随机种子

### 挑战 6: 模型保存与结果记录 (Model Persistence and Result Logging)

**问题描述：**
- 需要保存多个实验的结果进行对比
- 可视化多条学习曲线
- 生成可复现的实验报告

**解决方案：**

1. **结构化结果存储**：
   - 使用 JSON 格式保存实验结果
   - 为每个配置记录：
     - 最终准确率 (final_accuracy)
     - 测试损失 (test_loss)
     - 训练时间 (training_time)
     - 完整训练历史 (history)
   - 示例文件: `experiment_results.json`

2. **自动化可视化**：
   - 生成训练曲线图（准确率、损失）
   - 保存为高分辨率图像（300 DPI）
   - 创建多实验对比图

3. **命名规范**：
   - 文件名包含配置信息：`fl_iid_10clients_E3_B1024.png`
   - 便于后期分析和报告撰写

---

---

## References

1. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In Artificial Intelligence and Statistics (pp. 1273-1282). PMLR.

2. Kairouz, P., McMahan, H. B., et al. (2019). Advances and open problems in federated learning. arXiv preprint arXiv:1912.00967.

3. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine Learning and Systems, 2, 429-450.

4. Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with non-iid data. arXiv preprint arXiv:1806.00582.

5. PyTorch Documentation. (2024). torch.nn - Neural Network Modules. Retrieved from https://pytorch.org/docs/stable/nn.html

6. PyTorch Documentation. (2024). torchvision.datasets - Fashion-MNIST. Retrieved from https://pytorch.org/vision/stable/datasets.html#fashion-mnist

---

## Conclusion

This project successfully implemented a federated learning system for Fashion-MNIST image classification, demonstrating the key principles of distributed machine learning while preserving data privacy. The experiments revealed important insights:

1. **IID Performance**: Federated learning can achieve performance very close to centralized learning under IID conditions - with 5 clients achieving 86.43% accuracy compared to centralized 86.76% (only 0.33 percentage points difference)
2. **Non-IID Challenge**: Non-IID data distribution has a catastrophic impact on model performance, with 10-client configuration achieving only 41.24% accuracy (a 43.77 percentage point drop from IID performance)
3. **Client Count Impact**: Increasing the number of clients reduces accuracy in both IID (86.43% → 82.05% for 5→20 clients) and Non-IID scenarios (57.32% → 35.86% for 5→20 clients)
4. **Privacy-Performance Trade-off**: The privacy benefits of FL come with acceptable performance trade-offs in IID scenarios (0.4-5.4% relative accuracy loss), but are prohibitively expensive in Non-IID scenarios (51.5% relative accuracy loss)

The implementation provides a foundation for understanding real-world federated learning challenges and highlights the critical importance of addressing data heterogeneity in distributed systems.

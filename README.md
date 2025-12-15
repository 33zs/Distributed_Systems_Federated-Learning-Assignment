# JC4001 Distributed Systems - Federated Learning Assignment

This project implements a federated learning system for image classification using the Fashion-MNIST dataset, comparing it with centralised learning approaches.

## Project Structure

```
fdhomework/
├── README.md                          # This file
├── report.md                          # Comprehensive project report
├── centralised_learning.py            # Part 2: Centralised baseline implementation
├── federated_learning.py             # Part 3: Federated learning implementation
├── run_experiments.py                # Main script to run all experiments
├── requirements.txt                  # Python dependencies
└── JC4001 Coursework Assignment_2025.pdf  # Assignment specification
```

## Requirements

- Python 3.8+
- PyTorch 2.x + torchvision
- NumPy
- Matplotlib

## Installation

使用仓库内的依赖文件安装（推荐）：

```bash
pip install -r requirements.txt
```

如果需要手动安装，请确保安装了 PyTorch（含 torchvision）、numpy、matplotlib。

## How to Run

### Option 1: Run All Experiments (推荐)

一次性运行所有实验并生成结果与图表：

```bash
python run_experiments.py
```

脚本会：
1. 跑集中式基线（第 2 部分）
2. 跑联邦学习 IID 不同客户端数 5/10/20（第 4A 部分）
3. 跑联邦学习 IID vs Non-IID（第 4B 部分）
4. 生成对比图与分析（第 5 部分）
5. 将结果保存为 JSON，并导出可视化

**预计耗时：** 15–30 分钟，取决于是否有 GPU。

### Option 2: 单独运行

**仅集中式训练：**
```bash
python centralised_learning.py
```

**联邦学习示例：**
```bash
python federated_learning.py
```

## Generated Outputs

After running the experiments, the following files will be generated:

1. **Plots:** (saved in `plots/` folder)
   - `plots/centralised_training.png` - Centralised learning accuracy/loss curves
   - `plots/experiment1_client_counts.png` - Comparison of 5, 10, 20 clients (IID)
   - `plots/experiment2_iid_vs_non_iid.png` - IID vs Non-IID comparison
   - `plots/comprehensive_comparison.png` - All methods compared

2. **Models:**
   - `models/centralised_model.pth` - 集中式模型
   - `models/fl_5_clients_iid.pth`, `models/fl_10_clients_iid.pth`, `models/fl_20_clients_iid.pth`, `models/fl_10_clients_non_iid.pth` - 各联邦实验的全局模型

3. **Results:**
   - `experiment_results.json` - Detailed numerical results from all experiments

## Code Structure

### centralised_learning.py

Implements Part 2 of the assignment:
- Loads Fashion-MNIST dataset
- Creates a 3-layer neural network (784-128-64-10)
- Trains on the full dataset
- Evaluates and plots results

### federated_learning.py

Implements Part 3 of the assignment:
- `FederatedClient` class: Simulates individual clients with local data
- `FederatedServer` class: Manages global model and performs FedAvg aggregation
- `partition_data_iid()`: Creates IID data partitions
- `partition_data_non_iid()`: Creates non-IID data partitions (class-based skew)
- `run_federated_learning()`: Main FL training loop

### run_experiments.py

Runs all experiments for Parts 4 and 5:
- Experiment 1: Varying number of clients (5, 10, 20)
- Experiment 2: IID vs Non-IID data distribution
- Generates comprehensive comparisons and visualizations
- Saves all results

## Model Architecture

All models use the same architecture for fair comparison:

```
Input Layer:    784 neurons (28×28 flattened images)
Hidden Layer 1: 128 neurons, ReLU activation
Hidden Layer 2: 64 neurons, ReLU activation
Output Layer:   10 neurons, Softmax activation

Optimizer: Adam (learning rate = 0.001)
Loss: CrossEntropyLoss
```

## Federated Learning Configuration

- **Communication Rounds:** 20
- **Local Epochs per Round:** 5
- **Batch Size:** 512
- **Aggregation Algorithm:** FedAvg (Federated Averaging)
- **Client Participation:** 100% (all clients participate in each round)

## Data Distribution Strategies

### IID (Independent and Identically Distributed)
- Dataset randomly shuffled
- Split equally among clients
- Each client has uniform class distribution

### Non-IID (Non-Independent and Identically Distributed)
- Each client receives data from only 2-3 classes (out of 10)
- Simulates realistic heterogeneous data scenarios
- Creates statistical heterogeneity across clients

## Assignment Parts Mapping

- **Part 1:** See `report.md` - Federated Learning summary (500 words)
- **Part 2:** `centralised_learning.py` - Centralised baseline
- **Part 3:** `federated_learning.py` - FL implementation
  - Part 3A: Client simulation and data partitioning
  - Part 3B: Local model training
  - Part 3C: FedAvg server aggregation
  - Part 3D: Communication rounds
- **Part 4:** `run_experiments.py` - Experiments
  - Part 4A: Impact of number of clients
  - Part 4B: IID vs Non-IID comparison
- **Part 5:** `run_experiments.py` - Performance comparison
- **Report:** `report.md` - Complete project report

## Key Results Summary

After running the experiments, you can find:

1. **Final accuracies** for all methods in the console output
2. **Detailed metrics** in `experiment_results.json`
3. **Visual comparisons** in the generated PNG files

## Troubleshooting

### Out of Memory Error
If you encounter memory issues:
- Reduce batch size in the training functions
- Reduce number of clients
- Close other applications

### Slow Training
- GPU 加速能显著提升速度
- 检查 PyTorch 是否识别到 GPU：
  ```python
  import torch
  print("CUDA available:", torch.cuda.is_available())
  if torch.cuda.is_available():
      print("GPU count:", torch.cuda.device_count())
      print("GPU 0:", torch.cuda.get_device_name(0))
  ```

### ImportError
- 确认依赖已安装：`pip install -r requirements.txt`
- 检查 Python 版本：`python --version`（应为 3.8+）

## References

1. McMahan, H. B., et al. (2017). Communication-efficient learning of deep networks from decentralized data.
2. Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist
3. PyTorch documentation: https://pytorch.org/docs/stable/index.html

## Author

**Student Name:** zishanxu
**Course:** JC4001 - Distributed Systems
**Institution:** University of Aberdeen
**Academic Year:** 2025-2026

## License

This project is submitted as coursework for JC4001 Distributed Systems course.

# Distributed Systems - Federated Learning Assignment

This project implements a federated learning (FL) system for Fashion-MNIST image classification, comparing federated and centralised approaches under various configurations (IID/Non-IID data, different client counts).

---

## 📁 Project Structure

```
fdhomework/
├── README.md                          # This file
├── config.py                          # Centralized configuration (hyperparameters, paths)
├── model_utils.py                     # Neural network model definition
├── centralised_learning.py            # Part 2: Centralised baseline
├── federated_learning.py             # Part 3: Federated learning (FedAvg)
├── run_experiments.py                # Main entry point - runs all experiments
├── requirements.txt                  # Python dependencies
├── models/                           # Saved model checkpoints
├── plots/                            # Generated visualizations
└── results/                          # Experiment results (JSON)
```

---

## 🚀 Quick Start

### Installation

**Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install torch torchvision numpy matplotlib
```

**Requirements:**
- Python 3.8+
- PyTorch 2.x + torchvision
- NumPy
- Matplotlib

### Run All Experiments (Recommended)

**Default (auto GPU/CPU):**
```bash
python run_experiments.py
```

**Force CPU mode:**
```bash
python run_experiments.py --cpu
```

**Specify GPU device (Linux/Mac):**
```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py
```

**Expected runtime:** ~37 minutes with GPU (NVIDIA RTX 4090) | Longer on CPU

---

## 📊 What Gets Run

The `run_experiments.py` script executes all assignment parts:

1. **Part 2:** Centralised learning baseline (Fashion-MNIST, 15 epochs)
2. **Part 4A - Experiment 1:** FL with varying client counts (5, 10, 20 clients, IID data)
3. **Part 4B - Experiment 2:** FL with IID vs Non-IID data (5, 10, 20 clients each)
4. **Part 5:** Comprehensive performance comparison
5. **Outputs:** Models saved, plots generated, results stored as JSON

---

## 📂 Generated Outputs

After running, you'll find:

### 1. Plots (`plots/` folder)
- `centralised_training.png` - Centralised baseline training curves
- `experiment1_client_counts.png` - IID client count comparison (5/10/20)
- `experiment2_iid_vs_non_iid.png` - IID vs Non-IID comparison (10 clients)
- `experiment3_convergence_comparison.png` - Convergence speed analysis
- `comprehensive_comparison.png` - All configurations compared (6 subplots)

### 2. Models (`models/` folder)
- `centralised_model.pth` - Centralised baseline
- `fl_5_clients_iid.pth`, `fl_10_clients_iid.pth`, `fl_20_clients_iid.pth` - IID FL models
- `fl_5_clients_non_iid.pth`, `fl_10_clients_non_iid.pth`, `fl_20_clients_non_iid.pth` - Non-IID FL models

### 3. Results (`results/` folder)
- `experiment_results.json` - All metrics (accuracy, loss, time, convergence rounds)
- `system_info.json` - Hardware/software environment
- `config.json` - Experiment configuration snapshot

---

## 🧪 Experimental Configuration

### Model Architecture (All Experiments)
```
Input:    784 neurons (28×28 flattened Fashion-MNIST images)
Hidden 1: 128 neurons + ReLU
Hidden 2: 64 neurons + ReLU
Output:   10 neurons (10 classes)

Optimizer: Adam (lr=0.001)
Loss:      CrossEntropyLoss
```

### Centralised Learning
- **Epochs:** 15
- **Batch size:** 1024
- **Dataset:** Full 60,000 training samples
- **Result:** 86.76% test accuracy

### Federated Learning
- **Communication rounds (T):** 20
- **Local epochs per round (E):** 3
- **Batch size:** 1024
- **Aggregation:** FedAvg (weighted by client data size)
- **Client participation:** 100% (all clients every round)

### Data Distribution Strategies

**IID (Independent and Identically Distributed):**
- Dataset randomly shuffled and split equally
- Each client has uniform class distribution (all 10 classes)
- **Best result:** 86.43% (5 clients) - very close to centralised

**Non-IID (Non-Independent and Identically Distributed):**
- Each client receives only 2 classes (out of 10)
- Simulates realistic data heterogeneity (e.g., different hospitals, user preferences)
- **Challenge:** Severe performance drop (41.24% for 10 clients)

---

## 📈 Key Results Summary

| Configuration | Test Accuracy | Test Loss | Training Time |
|---------------|---------------|-----------|---------------|
| **Centralised** | **86.76%** | 0.3692 | 25.70s |
| FL (5 clients, IID) | 86.43% | 0.3821 | 173.24s |
| FL (10 clients, IID) | 85.01% | 0.4241 | 301.52s |
| FL (20 clients, IID) | 82.05% | 0.5133 | 589.84s |
| FL (5 clients, Non-IID) | 57.32% | 1.1860 | 173.55s |
| FL (10 clients, Non-IID) | **41.24%** | 2.9292 | 307.08s |
| FL (20 clients, Non-IID) | **35.86%** | 2.2074 | 583.60s |

**Key insights:**
- IID FL achieves near-centralised performance (0.33-4.71 pp drop)
- Non-IID data causes catastrophic degradation (43.77 pp drop for 10 clients)
- Client count affects performance: more clients = lower accuracy (especially Non-IID)

---

## 🔧 Code Structure Details

### `config.py`
Centralized configuration file containing all hyperparameters:
- Random seed (42)
- Model architecture sizes
- Training parameters (epochs, batch size, learning rate)
- FL parameters (rounds, local epochs)
- Directory paths

### `model_utils.py`
- `FashionMNISTNet` class: 3-layer fully connected network
- Shared across centralised and federated experiments

### `centralised_learning.py` (Part 2)
- `load_fashion_mnist()`: Dataset loading
- `train_centralised_model()`: Standard training loop
- `plot_training_history()`: Visualize training curves
- Serves as performance baseline

### `federated_learning.py` (Part 3)
- `FederatedClient`: Simulates individual clients with local data and training
- `FederatedServer`: Manages global model, performs FedAvg aggregation
- `partition_data_iid()`: Creates IID partitions (stratified sampling)
- `partition_data_non_iid()`: Creates Non-IID partitions (2 classes per client)
- `run_federated_learning()`: Main FL training loop (20 rounds)

### `run_experiments.py` (Parts 4 & 5)
- Orchestrates all experiments
- Runs 7 configurations: 1 centralised + 6 FL (3 IID + 3 Non-IID)
- Generates all plots and comparison figures
- Saves results to JSON

---

## 🐛 Troubleshooting

### Out of Memory Error
- **Solution 1:** Force CPU mode: `python run_experiments.py --cpu`
- **Solution 2:** Reduce batch size in `config.py` (line 18-26)
- **Solution 3:** Reduce number of clients or rounds

### Slow Training
Check if GPU is detected:
```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If no GPU, expect longer runtime (~2-3 hours on CPU).

### ImportError
```bash
pip install --upgrade torch torchvision numpy matplotlib
python --version  # Should be 3.8+
```

### Results Don't Match Report
- Ensure random seed is 42 (set in `config.py`)
- Verify configuration hasn't been modified
- Check all dependencies are correct versions

---

## 📚 References

1. McMahan, H. B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
2. Kairouz, P., et al. (2019). Advances and open problems in federated learning. *arXiv:1912.00967*.
3. Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist
4. PyTorch documentation: https://pytorch.org/docs/stable/


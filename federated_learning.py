"""
Part 3: Federated Learning Implementation
Implements a federated learning system with client simulation, FedAvg aggregation,
and support for both IID and non-IID data distributions using PyTorch.
"""

import os
import sys

# GPU/CPU Configuration
# To force CPU mode, run with: python federated_learning.py --cpu
# To use specific GPU, set environment variable: CUDA_VISIBLE_DEVICES=0 python federated_learning.py
USE_CPU = '--cpu' in sys.argv or os.environ.get('FORCE_CPU', '0') == '1'

if USE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("Running in CPU mode (forced)")
elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Default to GPU 0 if available

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict
import copy
from model_utils import create_model, get_model_weights, set_model_weights
from config import (
    RANDOM_SEED, BATCH_SIZE, LEARNING_RATE, FL_NUM_ROUNDS,
    FL_LOCAL_EPOCHS, NUM_WORKERS, OPTIMIZER_TYPE
)

plt.switch_backend('Agg')  # Use non-interactive backend


class FederatedClient:
    """
    Represents a client in the federated learning system.
    Each client has local data and can train a model independently.
    """

    def __init__(self, client_id: int, train_data: torch.Tensor, train_labels: torch.Tensor,
                 batch_size: int = BATCH_SIZE, learning_rate: float = LEARNING_RATE,
                 device: torch.device = None):
        """
        Initialize a federated client.

        Args:
            client_id: Unique identifier for this client
            train_data: Training data tensor
            train_labels: Training labels tensor
            batch_size: Batch size for local training (default from config)
            learning_rate: Learning rate for optimizer (default from config)
            device: Device to train on
        """
        self.client_id = client_id
        self.device = device if device else torch.device('cpu')
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Create dataset and dataloader
        # Note: num_workers from config for faster data loading
        self.dataset = TensorDataset(train_data, train_labels)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS
        )
        self.num_samples = len(self.dataset)

        # Model will be set by server
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def set_model(self, model: nn.Module):
        """
        Set the local model (typically receives from server).

        Args:
            model: PyTorch model
        """
        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, epochs: int = 1, verbose: int = 0):
        """
        Train the local model on local data.

        Args:
            epochs: Number of local training epochs
            verbose: Verbosity level
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            if verbose > 1:
                epoch_loss = running_loss / len(self.train_loader)
                epoch_acc = 100. * correct / total
                print(f"  Client {self.client_id} Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

    def get_weights(self):
        """
        Get current model weights.

        Returns:
            List of weight tensors
        """
        return get_model_weights(self.model)


class FederatedServer:
    """
    Represents the server in the federated learning system.
    Manages the global model and aggregates client updates.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize the federated server.

        Args:
            model: Initial global model
            device: Device to run on
        """
        self.device = device if device else torch.device('cpu')
        self.global_model = model.to(self.device)

    def get_global_model(self):
        """
        Get a copy of the global model.

        Returns:
            Copy of global model
        """
        return copy.deepcopy(self.global_model)

    def federated_averaging(self, client_weights: List[List[torch.Tensor]],
                           client_sizes: List[int]) -> List[torch.Tensor]:
        """
        Aggregate client models using FedAvg algorithm.

        Args:
            client_weights: List of client weight lists
            client_sizes: List of client dataset sizes

        Returns:
            Aggregated weights
        """
        total_size = sum(client_sizes)
        aggregated_weights = []

        # For each parameter
        num_params = len(client_weights[0])
        for param_idx in range(num_params):
            # Weighted average
            weighted_param = torch.zeros_like(client_weights[0][param_idx])
            for client_idx, client_weight_list in enumerate(client_weights):
                weight = client_sizes[client_idx] / total_size
                weighted_param += weight * client_weight_list[param_idx]

            aggregated_weights.append(weighted_param)

        return aggregated_weights

    def update_global_model(self, aggregated_weights: List[torch.Tensor]):
        """
        Update the global model with aggregated weights.

        Args:
            aggregated_weights: Aggregated weights from clients
        """
        set_model_weights(self.global_model, aggregated_weights)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the global model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Tuple of (loss, accuracy)
        """
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy


def partition_data_iid(dataset, num_clients: int) -> List[Subset]:
    """
    Partition data into IID (Independent and Identically Distributed) subsets.
    Uses stratified sampling to ensure each client gets uniform class distribution.

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients

    Returns:
        List of Subset objects for each client
    """
    num_classes = 10
    data_per_client = len(dataset) // num_clients

    # Group indices by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Shuffle within each class
    for class_id in range(num_classes):
        np.random.shuffle(class_indices[class_id])

    # Distribute evenly to each client
    client_indices = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        samples_per_client = len(class_indices[class_id]) // num_clients

        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:
                end_idx = len(class_indices[class_id])
            else:
                end_idx = (i + 1) * samples_per_client

            client_indices[i].extend(class_indices[class_id][start_idx:end_idx])

    # Shuffle each client's indices
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    # Create Subset objects
    client_datasets = [Subset(dataset, indices) for indices in client_indices]

    return client_datasets


def partition_data_non_iid(dataset, num_clients: int, classes_per_client: int = 2) -> List[Subset]:
    """
    Partition data into non-IID subsets with strict class distribution control.
    Each client gets data from exactly classes_per_client classes.
    GUARANTEES all classes are assigned to at least one client.

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        classes_per_client: Number of classes each client should have

    Returns:
        List of Subset objects for each client
    """
    num_classes = 10

    # Group indices by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Shuffle within each class
    for class_id in range(num_classes):
        np.random.shuffle(class_indices[class_id])

    # Improved class assignment strategy to guarantee all classes are covered
    # while respecting classes_per_client constraint
    class_assignment = [set() for _ in range(num_clients)]

    # Step 1: Ensure each class is assigned to at least one client
    # Strategy: Distribute classes evenly first, then randomize
    shuffled_classes = list(range(num_classes))
    np.random.shuffle(shuffled_classes)

    # Calculate total slots available
    total_slots = num_clients * classes_per_client

    if total_slots < num_classes:
        raise ValueError(f"Cannot distribute {num_classes} classes to {num_clients} clients "
                        f"with {classes_per_client} classes per client. "
                        f"Need at least {num_classes // classes_per_client} clients.")

    # Assign each class to a random client that has room
    for class_id in shuffled_classes:
        # Find clients that haven't reached their limit
        available_clients = [i for i in range(num_clients)
                           if len(class_assignment[i]) < classes_per_client]

        if not available_clients:
            raise RuntimeError("No available clients for class assignment. This should not happen.")

        # Randomly choose one available client
        client_id = np.random.choice(available_clients)
        class_assignment[client_id].add(class_id)

    # Step 2: Fill remaining slots to reach exactly classes_per_client for each client
    # Create a pool of all classes for random selection
    for client_id in range(num_clients):
        current_classes = class_assignment[client_id]

        # If this client needs more classes
        while len(current_classes) < classes_per_client:
            # Get available classes (not yet assigned to this client)
            available_classes = [c for c in range(num_classes) if c not in current_classes]

            if len(available_classes) == 0:
                # If no more classes available (shouldn't happen with classes_per_client <= num_classes)
                break

            # Randomly select one more class
            new_class = np.random.choice(available_classes)
            current_classes.add(new_class)

    # Verification: Check that all classes are covered
    covered_classes = set()
    for client_classes in class_assignment:
        covered_classes.update(client_classes)

    if len(covered_classes) != num_classes:
        missing_classes = set(range(num_classes)) - covered_classes
        raise RuntimeError(f"Class coverage check failed! Missing classes: {missing_classes}")

    # Distribute data
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        clients_with_class = [i for i in range(num_clients) if class_id in class_assignment[i]]

        # This should never happen now, but keep as safety check
        if len(clients_with_class) == 0:
            raise RuntimeError(f"No client assigned to class {class_id}! This should not happen.")

        indices = class_indices[class_id]
        samples_per_client = len(indices) // len(clients_with_class)

        for idx, client_id in enumerate(clients_with_class):
            start_idx = idx * samples_per_client
            if idx == len(clients_with_class) - 1:
                end_idx = len(indices)
            else:
                end_idx = (idx + 1) * samples_per_client

            client_indices[client_id].extend(indices[start_idx:end_idx])

    # Shuffle each client's indices
    for i in range(num_clients):
        if len(client_indices[i]) > 0:
            np.random.shuffle(client_indices[i])

    # Create Subset objects
    client_datasets = [Subset(dataset, indices) if len(indices) > 0 else None
                       for indices in client_indices]

    # Check for empty clients
    for i, dataset_subset in enumerate(client_datasets):
        if dataset_subset is None:
            raise ValueError(f"Client {i} has no data assigned")

    return client_datasets


def load_and_preprocess_data(batch_size: int = BATCH_SIZE):
    """
    Load and preprocess Fashion-MNIST dataset using torchvision.

    Args:
        batch_size: Batch size for test loader (default from config)

    Returns:
        Tuple of (train_dataset, test_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor()
        # Note: Flattening is handled in model's forward() method
    ])

    # Load datasets
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Note: num_workers from config for faster data loading
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_dataset, test_loader


def run_federated_learning(num_clients: int,
                          num_rounds: int = FL_NUM_ROUNDS,
                          local_epochs: int = FL_LOCAL_EPOCHS,
                          batch_size: int = BATCH_SIZE,
                          learning_rate: float = LEARNING_RATE,
                          data_distribution: str = 'iid',
                          verbose: int = 1):
    """
    Run the federated learning training process.

    Args:
        num_clients: Number of clients to simulate
        num_rounds: Number of communication rounds (default from config)
        local_epochs: Number of local training epochs per round (default from config)
        batch_size: Batch size for local training (default from config)
        learning_rate: Learning rate (default from config)
        data_distribution: 'iid' or 'non_iid'
        verbose: Verbosity level

    Returns:
        Tuple of (model, history)
    """
    print("=" * 70)
    print(f"FEDERATED LEARNING - {data_distribution.upper()} with {num_clients} clients")
    print("=" * 70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load data
    try:
        train_dataset, test_loader = load_and_preprocess_data(batch_size)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        raise

    print(f"\nDataset: Fashion-MNIST")
    print(f"  - Total classes: 10")
    print(f"  - Training samples: {len(train_dataset):,}")
    print(f"  - Test samples: {len(test_loader.dataset):,}")
    print(f"  - Image size: 28x28")

    # Partition data
    print(f"\nPartitioning data ({data_distribution})...")
    if data_distribution == 'iid':
        client_datasets = partition_data_iid(train_dataset, num_clients)
    else:
        client_datasets = partition_data_non_iid(train_dataset, num_clients)

    # Create clients
    clients = []
    print(f"\nClient Data Distribution:")
    print("-" * 70)

    for i, client_dataset in enumerate(client_datasets):
        # Extract data and labels
        indices = client_dataset.indices
        data_list = []
        labels_list = []

        for idx in indices:
            data, label = train_dataset[idx]
            data_list.append(data)
            labels_list.append(label)

        train_data = torch.stack(data_list)
        train_labels = torch.tensor(labels_list)

        # Create client
        client = FederatedClient(
            i, train_data, train_labels,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )
        clients.append(client)

        if verbose:
            # Calculate class distribution
            class_counts = {}
            for label in labels_list:
                label = int(label)
                class_counts[label] = class_counts.get(label, 0) + 1

            class_dist_str = ", ".join([f"{c}:{cnt}" for c, cnt in sorted(class_counts.items())])
            print(f"Client {i}: {client.num_samples:5d} samples | Classes: [{class_dist_str}]")

    # Print summary
    total_samples = sum([c.num_samples for c in clients])
    avg_samples = total_samples / num_clients
    print("-" * 70)
    print(f"Total distributed samples: {total_samples:,}")
    print(f"Average samples per client: {avg_samples:.0f}")
    print("-" * 70)

    # Create server
    print("\nInitializing server with global model...")
    global_model = create_model()
    server = FederatedServer(global_model, device=device)

    # Training history
    history = {
        'rounds': [],
        'test_accuracy': [],
        'test_loss': []
    }

    # Initial evaluation
    test_loss, test_accuracy = server.evaluate(test_loader)
    print(f"\nInitial - Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")

    # Federated learning rounds
    print(f"\nStarting {num_rounds} communication rounds...")
    start_time = time.time()

    for round_num in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"{'='*70}")

        # Step 1: Broadcast global model to all clients
        for client in clients:
            client.set_model(server.get_global_model())

        # Step 2: Each client trains locally
        if verbose:
            print("Clients training locally...")
        for client in clients:
            client.train(epochs=local_epochs, verbose=verbose)

        # Step 3: Collect model updates from clients
        client_weights = []
        client_sizes = []
        for client in clients:
            client_weights.append(client.get_weights())
            client_sizes.append(client.num_samples)

        # Step 4: Server aggregates models using FedAvg
        if verbose:
            print("Server aggregating models...")
        aggregated_weights = server.federated_averaging(client_weights, client_sizes)

        # Step 5: Update global model
        server.update_global_model(aggregated_weights)

        # Evaluate global model on test set
        test_loss, test_accuracy = server.evaluate(test_loader)

        history['rounds'].append(round_num + 1)
        history['test_accuracy'].append(test_accuracy / 100.0)  # Store as fraction
        history['test_loss'].append(test_loss)

        print(f"\nRound {round_num + 1} Results:")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}")

    training_time = time.time() - start_time

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Final Test Accuracy: {history['test_accuracy'][-1]*100:.2f}%")
    print(f"Final Test Loss: {history['test_loss'][-1]:.4f}")
    print(f"Total Training Time: {training_time:.2f} seconds")
    print("=" * 70)

    history['training_time'] = training_time
    history['num_clients'] = num_clients
    history['data_distribution'] = data_distribution

    return server.get_global_model(), history


def plot_federated_results(histories: Dict[str, dict], save_path: str):
    """
    Plot federated learning results for multiple experiments.
    Shows accuracy and loss over communication rounds.

    Args:
        histories: Dictionary mapping experiment names to history dictionaries
        save_path: Path to save the plot
    """
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy over rounds
    for name, history in histories.items():
        ax1.plot(history['rounds'],
                [acc * 100 for acc in history['test_accuracy']],
                marker='o', label=name, linewidth=2)

    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Federated Learning - Accuracy over Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss over rounds
    for name, history in histories.items():
        ax2.plot(history['rounds'], history['test_loss'],
                marker='s', label=name, linewidth=2)

    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Federated Learning - Loss over Rounds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    """Main function for testing federated learning."""
    # Note: Random seeds are set here for standalone execution.
    # When called from run_experiments.py, seeds are set there.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Run federated learning with config parameters
    model, history = run_federated_learning(
        num_clients=10,
        num_rounds=FL_NUM_ROUNDS,
        local_epochs=FL_LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        data_distribution='iid',
        verbose=1
    )

    # Plot results
    histories = {'10 Clients (IID)': history}
    plot_federated_results(histories, 'plots/federated_demo.png')

    print("\nFederated learning demo completed!")


if __name__ == "__main__":
    main()

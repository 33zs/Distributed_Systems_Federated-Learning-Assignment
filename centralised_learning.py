"""
Part 2: Centralised Learning Baseline
Implements centralized training on Fashion-MNIST dataset as a baseline for comparison.
Uses PyTorch and torchvision.
"""

import os
import sys

# GPU/CPU Configuration
# To force CPU mode, run with: python centralised_learning.py --cpu
# To use specific GPU, set environment variable: CUDA_VISIBLE_DEVICES=0 python centralised_learning.py
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from typing import Tuple, Dict, List
from model_utils import create_model
from config import (
    RANDOM_SEED, BATCH_SIZE, LEARNING_RATE, CENTRALISED_EPOCHS,
    NUM_WORKERS, OPTIMIZER_TYPE
)

plt.switch_backend('Agg')  # Use non-interactive backend


def load_fashion_mnist(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess the Fashion-MNIST dataset using torchvision.

    Args:
        batch_size: Batch size for data loaders (default from config)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transforms: Convert to tensor (flattening will be done in model)
    transform = transforms.Compose([
        transforms.ToTensor()
        # Note: Flattening is handled in model's forward() method
    ])

    # Download and load training dataset
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Download and load test dataset
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    # Note: num_workers from config for faster data loading
    # Set to 0 if you encounter multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, test_loader


def train_epoch(model: nn.Module, train_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, test_loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the model on test data.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Statistics
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_centralised_model(epochs: int = CENTRALISED_EPOCHS,
                            batch_size: int = BATCH_SIZE,
                            learning_rate: float = LEARNING_RATE,
                            verbose: int = 1) -> Tuple[nn.Module, Dict, float]:
    """
    Train a centralized model on the Fashion-MNIST dataset.

    Args:
        epochs: Number of training epochs (default from config)
        batch_size: Batch size for training (default from config)
        learning_rate: Learning rate for optimizer (default from config)
        verbose: Verbosity mode (0, 1, or 2)

    Returns:
        Tuple of (model, history, training_time)
    """
    print("=" * 60)
    print("CENTRALISED LEARNING BASELINE")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load data
    print("\nLoading Fashion-MNIST dataset...")
    try:
        train_loader, test_loader = load_fashion_mnist(batch_size)
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        raise

    # Create model
    print("\nCreating neural network model...")
    model = create_model().to(device)

    if verbose:
        print(f"Model architecture:")
        print(model)
        from model_utils import count_parameters
        print(f"Total parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    # Training loop
    print(f"\nTraining model for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    training_time = time.time() - start_time

    # Final evaluation
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final Training Accuracy: {history['train_accuracy'][-1]:.2f}%")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print("=" * 60)

    return model, history, training_time


def plot_training_history(history: Dict, save_path: str = 'plots/centralised_training.png') -> None:
    """
    Plot training accuracy/loss curves.

    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)

    print(f"\nGenerating training curves plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_accuracy']) + 1)

    # Plot accuracy
    ax1.plot(epochs, history['train_accuracy'], label='Training Accuracy', marker='o', linewidth=2)
    ax1.plot(epochs, history['test_accuracy'], label='Test Accuracy', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Centralised Learning - Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(epochs, history['train_loss'], label='Training Loss', marker='o', linewidth=2)
    ax2.plot(epochs, history['test_loss'], label='Test Loss', marker='s', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Centralised Learning - Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def main() -> Tuple[nn.Module, Dict]:
    """Main function to run centralised learning baseline."""
    # Note: Random seeds are set here for standalone execution.
    # When called from run_experiments.py, seeds are set there.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Train centralised model with config parameters
    model, history, training_time = train_centralised_model(
        epochs=CENTRALISED_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/centralised_model.pth')
    print("\nModel saved to: models/centralised_model.pth")

    return model, history


if __name__ == "__main__":
    main()

"""
Main Experiment Runner
Runs all experiments for Parts 2, 3, 4, and 5 of the assignment.
"""

import os
import sys

# GPU/CPU Configuration
# To force CPU mode, run with: python run_experiments.py --cpu
# To use specific GPU, set environment variable: CUDA_VISIBLE_DEVICES=0 python run_experiments.py
USE_CPU = '--cpu' in sys.argv or os.environ.get('FORCE_CPU', '0') == '1'

if USE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("Running in CPU mode (forced)")
elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Default to GPU 0 if available

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime

plt.switch_backend('Agg')  # Use non-interactive backend

# Import from our modules
from centralised_learning import train_centralised_model, load_fashion_mnist, plot_training_history
from federated_learning import run_federated_learning, plot_federated_results
from config import (
    RANDOM_SEED, BATCH_SIZE, LEARNING_RATE, CENTRALISED_EPOCHS,
    FL_NUM_ROUNDS, FL_LOCAL_EPOCHS, FL_CLIENT_COUNTS,
    OPTIMIZER_TYPE, LOSS_FUNCTION, NUM_CLASSES, INPUT_SIZE,
    HIDDEN_SIZE_1, HIDDEN_SIZE_2, PLOTS_DIR, MODELS_DIR, RESULTS_DIR
)


def calculate_convergence_round(history, threshold_ratio=0.95):
    """
    Calculate the number of rounds needed to reach target accuracy.

    Args:
        history: Training history dictionary
        threshold_ratio: Target accuracy as ratio of final accuracy (default 0.95)

    Returns:
        Number of rounds needed to converge
    """
    test_accuracy = history['test_accuracy']
    final_accuracy = test_accuracy[-1]
    target_accuracy = final_accuracy * threshold_ratio

    for round_idx, acc in enumerate(test_accuracy):
        if acc >= target_accuracy:
            return round_idx + 1  # Round numbers start from 1

    return len(test_accuracy)  # If not reached, return total rounds


def save_system_info(filepath: str):
    """
    Save system and environment information.

    Args:
        filepath: Path to save the system info JSON
    """
    import platform
    import sys

    system_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'cuda': {
            'available': torch.cuda.is_available(),
            'version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }

    # Add GPU info if available
    if torch.cuda.is_available():
        system_info['cuda']['devices'] = []
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                'compute_capability': f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
            }
            system_info['cuda']['devices'].append(device_info)

    try:
        with open(filepath, 'w') as f:
            json.dump(system_info, f, indent=2)
        print(f"[OK] System info saved to: {filepath}")
    except Exception as e:
        print(f"Warning: Failed to save system info: {e}")


def save_config_info(filepath: str):
    """
    Save experiment configuration and hyperparameters from config.py.

    Args:
        filepath: Path to save the config JSON
    """
    config = {
        'experiment_name': 'JC4001 Federated Learning',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': RANDOM_SEED,
        'dataset': {
            'name': 'Fashion-MNIST',
            'num_classes': NUM_CLASSES,
            'input_size': INPUT_SIZE,
            'train_samples': 60000,
            'test_samples': 10000,
        },
        'model_architecture': {
            'type': 'FashionMNISTNet',
            'layers': [
                {'type': 'Linear', 'input': INPUT_SIZE, 'output': HIDDEN_SIZE_1, 'activation': 'ReLU'},
                {'type': 'Linear', 'input': HIDDEN_SIZE_1, 'output': HIDDEN_SIZE_2, 'activation': 'ReLU'},
                {'type': 'Linear', 'input': HIDDEN_SIZE_2, 'output': NUM_CLASSES, 'activation': 'None'},
            ],
            'total_parameters': 109386,
        },
        'centralised_learning': {
            'epochs': CENTRALISED_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer': OPTIMIZER_TYPE,
            'loss_function': LOSS_FUNCTION,
        },
        'federated_learning': {
            'communication_rounds': FL_NUM_ROUNDS,
            'local_epochs': FL_LOCAL_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer': OPTIMIZER_TYPE,
            'aggregation_method': 'FedAvg',
            'client_participation_rate': 1.0,
        },
        'experiments': {
            'part2': {
                'name': 'Centralised Learning Baseline',
                'description': 'Train centralized model on full dataset'
            },
            'part4a': {
                'name': 'Experiment 1 - Client Count Impact',
                'description': 'Compare FL performance with different client counts (IID)',
                'client_counts': FL_CLIENT_COUNTS,
                'data_distribution': 'IID'
            },
            'part4b': {
                'name': 'Experiment 2 - IID vs Non-IID',
                'description': 'Compare IID and Non-IID data distributions',
                'client_counts': FL_CLIENT_COUNTS,
                'data_distributions': ['IID', 'Non-IID'],
                'non_iid_classes_per_client': 2
            }
        }
    }

    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[OK] Configuration saved to: {filepath}")
    except Exception as e:
        print(f"Warning: Failed to save config: {e}")


def run_all_experiments():
    """
    Run all experiments for the assignment and generate comprehensive results.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "JC4001 FEDERATED LEARNING EXPERIMENTS")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Set random seeds for reproducibility (from config)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Create output directories (from config)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    results = {}

    # Save system and configuration info
    save_system_info('results/system_info.json')
    save_config_info('results/config.json')

    # =========================================================================
    # PART 2: Centralised Learning Baseline
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# PART 2: CENTRALISED LEARNING BASELINE")
    print("#" * 80)

    centralised_model, centralised_history, centralised_time = train_centralised_model(
        epochs=CENTRALISED_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        verbose=1
    )

    # Get final results from history (already evaluated during training)
    central_acc = centralised_history['test_accuracy'][-1]
    central_loss = centralised_history['test_loss'][-1]

    results['centralised'] = {
        'final_accuracy': central_acc,
        'final_loss': central_loss,
        'training_time': centralised_time,
        'epochs': CENTRALISED_EPOCHS,
        'history': {
            'accuracy': centralised_history['train_accuracy'],
            'loss': centralised_history['train_loss']
        }
    }

    # Save centralised model and training curves
    print("\n" + "=" * 60)
    print("SAVING MODEL AND PLOTS")
    print("=" * 60)
    try:
        print("Creating models directory...")
        os.makedirs('models', exist_ok=True)
        print("Saving centralised model (this may take a few seconds)...")
        torch.save(centralised_model.state_dict(), 'models/centralised_model.pth')
        print("[OK] Centralised model saved to: models/centralised_model.pth")
    except Exception as e:
        print(f"Warning: Failed to save centralised model: {e}")

    try:
        os.makedirs('plots', exist_ok=True)
        plot_training_history(centralised_history, 'plots/centralised_training.png')
        print("[OK] Centralised training plot saved to: plots/centralised_training.png")
    except Exception as e:
        print(f"Warning: Failed to save training plot: {e}")

    print("=" * 60)

    # =========================================================================
    # PART 3 & 4A: Federated Learning with Different Numbers of Clients (IID)
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# PART 4A: EXPERIMENT 1 - IMPACT OF NUMBER OF CLIENTS (IID)")
    print("#" * 80)

    client_counts = FL_CLIENT_COUNTS
    fl_histories_clients = {}

    for num_clients in client_counts:
        print(f"\n{'='*80}")
        print(f" Running FL with {num_clients} clients (IID)")
        print(f"{'='*80}")

        model, history = run_federated_learning(
            num_clients=num_clients,
            num_rounds=FL_NUM_ROUNDS,
            local_epochs=FL_LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            data_distribution='iid',
            verbose=1
        )

        exp_name = f'{num_clients} Clients (IID)'
        fl_histories_clients[exp_name] = history

        # Calculate convergence round
        convergence_round = calculate_convergence_round(history, threshold_ratio=0.95)

        results[f'fl_iid_{num_clients}_clients'] = {
            'final_accuracy': history['test_accuracy'][-1],
            'final_loss': history['test_loss'][-1],
            'training_time': history['training_time'],
            'num_clients': num_clients,
            'rounds': FL_NUM_ROUNDS,
            'convergence_round': convergence_round
        }

        # Save FL global model
        try:
            model_path = f'models/fl_{num_clients}_clients_iid.pth'
            torch.save(model.state_dict(), model_path)
            print(f"[OK] FL model saved to: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to save FL model: {e}")

    # Plot comparison for different client counts
    print("\nGenerating experiment 1 plot (client counts comparison)...")
    try:
        os.makedirs('plots', exist_ok=True)
        plot_federated_results(
            fl_histories_clients,
            'plots/experiment1_client_counts.png'
        )
        print("[OK] Experiment 1 plot saved to: plots/experiment1_client_counts.png")
    except Exception as e:
        print(f"Warning: Failed to save experiment1 plot: {e}")

    # =========================================================================
    # PART 4B: Federated Learning - IID vs Non-IID (Multiple Client Counts)
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# PART 4B: EXPERIMENT 2 - IID vs NON-IID DATA DISTRIBUTION")
    print("#" * 80)

    fl_histories_non_iid = {}

    # Run Non-IID experiments for 5, 10, 20 clients
    for num_clients in client_counts:
        print(f"\n{'='*80}")
        print(f" Running FL with {num_clients} clients (Non-IID)")
        print(f"{'='*80}")

        model_non_iid, history_non_iid = run_federated_learning(
            num_clients=num_clients,
            num_rounds=FL_NUM_ROUNDS,
            local_epochs=FL_LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            data_distribution='non_iid',
            verbose=1
        )

        exp_name = f'{num_clients} Clients (Non-IID)'
        fl_histories_non_iid[exp_name] = history_non_iid

        # Calculate convergence round
        convergence_round_non_iid = calculate_convergence_round(history_non_iid, threshold_ratio=0.95)

        results[f'fl_non_iid_{num_clients}_clients'] = {
            'final_accuracy': history_non_iid['test_accuracy'][-1],
            'final_loss': history_non_iid['test_loss'][-1],
            'training_time': history_non_iid['training_time'],
            'num_clients': num_clients,
            'rounds': FL_NUM_ROUNDS,
            'convergence_round': convergence_round_non_iid
        }

        # Save FL global model
        try:
            model_path = f'models/fl_{num_clients}_clients_non_iid.pth'
            torch.save(model_non_iid.state_dict(), model_path)
            print(f"[OK] FL model saved to: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to save FL model: {e}")

    # Plot IID vs Non-IID comparison
    print("\nGenerating experiment 2 plot (IID vs Non-IID comparison)...")
    try:
        os.makedirs('plots', exist_ok=True)
        # Combine IID and Non-IID histories for plotting
        fl_histories_combined = {}
        fl_histories_combined.update(fl_histories_clients)  # IID results
        fl_histories_combined.update(fl_histories_non_iid)  # Non-IID results

        plot_federated_results(
            fl_histories_combined,
            'plots/experiment2_iid_vs_non_iid.png'
        )
        print("[OK] Experiment 2 plot saved to: plots/experiment2_iid_vs_non_iid.png")
    except Exception as e:
        print(f"Warning: Failed to save experiment2 plot: {e}")

    # =========================================================================
    # EXPERIMENT 3: Convergence Speed Comparison (Standalone Plot)
    # =========================================================================
    print("\nGenerating convergence speed comparison plot...")
    try:
        os.makedirs('plots', exist_ok=True)

        # Create a standalone convergence plot
        fig_conv = plt.figure(figsize=(12, 7))
        ax_conv = fig_conv.add_subplot(111)

        # Prepare data
        methods_fl = ['FL (5, IID)', 'FL (10, IID)', 'FL (20, IID)',
                      'FL (5, Non-IID)', 'FL (10, Non-IID)', 'FL (20, Non-IID)']
        convergence_rounds = [
            results['fl_iid_5_clients']['convergence_round'],
            results['fl_iid_10_clients']['convergence_round'],
            results['fl_iid_20_clients']['convergence_round'],
            results['fl_non_iid_5_clients']['convergence_round'],
            results['fl_non_iid_10_clients']['convergence_round'],
            results['fl_non_iid_20_clients']['convergence_round']
        ]

        # Colors: IID in blue, Non-IID in red
        colors_fl = ['#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']

        # Create bar chart
        bars_conv = ax_conv.bar(methods_fl, convergence_rounds, color=colors_fl,
                                alpha=0.7, edgecolor='black', linewidth=1.5)

        # Styling
        ax_conv.set_ylabel('Rounds to Converge (95% of Final Accuracy)',
                          fontsize=14, fontweight='bold')
        ax_conv.set_xlabel('Federated Learning Configuration',
                          fontsize=14, fontweight='bold')
        ax_conv.set_title('Convergence Speed Comparison: IID vs Non-IID Data Distribution',
                         fontsize=16, fontweight='bold', pad=20)
        ax_conv.set_ylim([0, max(convergence_rounds) + 5])
        ax_conv.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add value labels on bars
        for bar, rounds in zip(bars_conv, convergence_rounds):
            height = bar.get_height()
            ax_conv.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rounds}',
                        ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.7, edgecolor='black', label='IID Data'),
            Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='Non-IID Data')
        ]
        ax_conv.legend(handles=legend_elements, fontsize=12, loc='upper left')

        # Rotate x-axis labels
        plt.setp(ax_conv.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)

        plt.tight_layout()
        plt.savefig('plots/experiment3_convergence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Convergence comparison plot saved to: plots/experiment3_convergence_comparison.png")
    except Exception as e:
        print(f"Warning: Failed to save convergence comparison plot: {e}")

    # =========================================================================
    # PART 5: Comprehensive Comparison
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# PART 5: COMPREHENSIVE COMPARISON")
    print("#" * 80)

    # Create comprehensive comparison plot
    print("\nGenerating comprehensive comparison plots (6 subplots)...")
    print("This may take a few seconds...")

    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    # Create figure with 2x3 layout for 6 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Final Accuracy
    ax2 = fig.add_subplot(gs[0, 1])  # Training Time
    ax3 = fig.add_subplot(gs[0, 2])  # Convergence Rounds (NEW)
    ax4 = fig.add_subplot(gs[1, 0])  # IID client counts
    ax5 = fig.add_subplot(gs[1, 1])  # Non-IID client counts
    ax6 = fig.add_subplot(gs[1, 2])  # IID vs Non-IID combined

    # Plot 1: Final Accuracy Comparison
    methods = ['Centralised',
               'FL (5, IID)', 'FL (10, IID)', 'FL (20, IID)',
               'FL (5, Non-IID)', 'FL (10, Non-IID)', 'FL (20, Non-IID)']
    # Note: centralised stores accuracy as percentage (0-100), FL stores as fraction (0-1)
    accuracies = [
        results['centralised']['final_accuracy'],  # Already percentage
        results['fl_iid_5_clients']['final_accuracy'] * 100,
        results['fl_iid_10_clients']['final_accuracy'] * 100,
        results['fl_iid_20_clients']['final_accuracy'] * 100,
        results['fl_non_iid_5_clients']['final_accuracy'] * 100,
        results['fl_non_iid_10_clients']['final_accuracy'] * 100,
        results['fl_non_iid_20_clients']['final_accuracy'] * 100
    ]

    colors = ['#2ecc71', '#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(accuracies) - 5, max(accuracies) + 2])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

    # Plot 2: Training Time Comparison
    times = [
        results['centralised']['training_time'],
        results['fl_iid_5_clients']['training_time'],
        results['fl_iid_10_clients']['training_time'],
        results['fl_iid_20_clients']['training_time'],
        results['fl_non_iid_5_clients']['training_time'],
        results['fl_non_iid_10_clients']['training_time'],
        results['fl_non_iid_20_clients']['training_time']
    ]

    bars2 = ax2.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, t in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

    # Plot 3: Convergence Rounds Comparison (NEW)
    methods_fl = ['FL (5, IID)', 'FL (10, IID)', 'FL (20, IID)',
                  'FL (5, Non-IID)', 'FL (10, Non-IID)', 'FL (20, Non-IID)']
    convergence_rounds = [
        results['fl_iid_5_clients']['convergence_round'],
        results['fl_iid_10_clients']['convergence_round'],
        results['fl_iid_20_clients']['convergence_round'],
        results['fl_non_iid_5_clients']['convergence_round'],
        results['fl_non_iid_10_clients']['convergence_round'],
        results['fl_non_iid_20_clients']['convergence_round']
    ]

    colors_fl = ['#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']
    bars3 = ax3.bar(methods_fl, convergence_rounds, color=colors_fl, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Rounds to Converge', fontsize=12, fontweight='bold')
    ax3.set_title('Convergence Speed (Rounds to 95% Final Accuracy)', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, max(convergence_rounds) + 3])
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, rounds in zip(bars3, convergence_rounds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rounds}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

    # Plot 4: FL Accuracy over rounds (IID - different client counts)
    for name, history in fl_histories_clients.items():
        ax4.plot(history['rounds'],
                [acc * 100 for acc in history['test_accuracy']],
                marker='o', label=name, linewidth=2)

    ax4.axhline(y=results['centralised']['final_accuracy'],
               color='#2ecc71', linestyle='--', linewidth=2,
               label='Centralised Baseline')

    ax4.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Impact of Number of Clients (IID)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: FL Accuracy over rounds (Non-IID - different client counts)
    for name, history in fl_histories_non_iid.items():
        ax5.plot(history['rounds'],
                [acc * 100 for acc in history['test_accuracy']],
                marker='s', label=name, linewidth=2)

    ax5.axhline(y=results['centralised']['final_accuracy'],
               color='#2ecc71', linestyle='--', linewidth=2,
               label='Centralised Baseline')

    ax5.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Impact of Number of Clients (Non-IID)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: IID vs Non-IID (Combined comparison)
    # Plot IID lines
    for name, history in fl_histories_clients.items():
        ax6.plot(history['rounds'],
                [acc * 100 for acc in history['test_accuracy']],
                marker='o', label=name, linewidth=2, linestyle='-', markersize=5)

    # Plot Non-IID lines
    for name, history in fl_histories_non_iid.items():
        ax6.plot(history['rounds'],
                [acc * 100 for acc in history['test_accuracy']],
                marker='s', label=name, linewidth=2, linestyle='--', markersize=5)

    ax6.axhline(y=results['centralised']['final_accuracy'],
               color='#2ecc71', linestyle=':', linewidth=2,
               label='Centralised Baseline')

    ax6.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax6.set_title('IID vs Non-IID Data Distribution (All Clients)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=8, loc='lower right')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        plt.savefig('plots/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        print("[OK] Comprehensive comparison plot saved to: plots/comprehensive_comparison.png")
    except Exception as e:
        print(f"\nWarning: Failed to save comprehensive comparison plot: {e}")
    finally:
        plt.close()

    # =========================================================================
    # Print Summary Table
    # =========================================================================
    print("\n\n" + "=" * 95)
    print(" " * 30 + "RESULTS SUMMARY TABLE")
    print("=" * 95)
    print(f"{'Method':<25} {'Accuracy':<15} {'Loss':<12} {'Time (s)':<12} {'Total Rounds':<15} {'Converge@95%':<15}")
    print("-" * 95)

    # Centralised stores accuracy as percentage already
    print(f"{'Centralised':<25} "
          f"{results['centralised']['final_accuracy']:>6.2f}% {'':<7} "
          f"{results['centralised']['final_loss']:>6.4f} {'':<5} "
          f"{results['centralised']['training_time']:>8.2f} {'':<3} "
          f"{results['centralised']['epochs']:>6} epochs {'':<6} "
          f"{'N/A':<15}")

    for clients in client_counts:
        key = f'fl_iid_{clients}_clients'
        print(f"{'FL ('+str(clients)+' clients, IID)':<25} "
              f"{results[key]['final_accuracy']*100:>6.2f}% {'':<7} "
              f"{results[key]['final_loss']:>6.4f} {'':<5} "
              f"{results[key]['training_time']:>8.2f} {'':<3} "
              f"{results[key]['rounds']:>6} rounds {'':<6} "
              f"{results[key]['convergence_round']:>6} rounds")

    for clients in client_counts:
        key = f'fl_non_iid_{clients}_clients'
        print(f"{'FL ('+str(clients)+' clients, Non-IID)':<25} "
              f"{results[key]['final_accuracy']*100:>6.2f}% {'':<7} "
              f"{results[key]['final_loss']:>6.4f} {'':<5} "
              f"{results[key]['training_time']:>8.2f} {'':<3} "
              f"{results[key]['rounds']:>6} rounds {'':<6} "
              f"{results[key]['convergence_round']:>6} rounds")

    print("=" * 95)

    # =========================================================================
    # Save Results to JSON
    # =========================================================================
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    results_serializable = convert_to_serializable(results)

    print("\nSaving experiment results to JSON file...")
    try:
        with open('results/experiment_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print("[OK] Detailed results saved to: results/experiment_results.json")
    except Exception as e:
        print(f"\nWarning: Failed to save experiment results JSON: {e}")

    # =========================================================================
    # Generate Insights
    # =========================================================================
    print("\n\n" + "=" * 80)
    print(" " * 30 + "KEY INSIGHTS")
    print("=" * 80)

    # Centralised stores as percentage, FL stores as fraction
    central_acc = results['centralised']['final_accuracy']  # Already percentage
    fl_iid_10_acc = results['fl_iid_10_clients']['final_accuracy'] * 100
    fl_non_iid_10_acc = results['fl_non_iid_10_clients']['final_accuracy'] * 100

    print(f"\n1. Centralised vs FL (IID):")
    print(f"   - Accuracy gap: {central_acc - fl_iid_10_acc:.2f}%")
    print(f"   - FL achieves {(fl_iid_10_acc/central_acc)*100:.1f}% of centralised performance")

    print(f"\n2. IID vs Non-IID (10 clients):")
    print(f"   - Accuracy drop with non-IID data: {fl_iid_10_acc - fl_non_iid_10_acc:.2f}%")
    print(f"   - Non-IID achieves {(fl_non_iid_10_acc/fl_iid_10_acc)*100:.1f}% of IID performance")

    print(f"\n3. Impact of Client Count (IID):")
    # FL results are stored as fractions, so multiply by 100
    acc_5_iid = results['fl_iid_5_clients']['final_accuracy'] * 100
    acc_10_iid = results['fl_iid_10_clients']['final_accuracy'] * 100
    acc_20_iid = results['fl_iid_20_clients']['final_accuracy'] * 100
    print(f"   - 5 clients:  {acc_5_iid:.2f}%")
    print(f"   - 10 clients: {acc_10_iid:.2f}%")
    print(f"   - 20 clients: {acc_20_iid:.2f}%")
    print(f"   - Change from 5 to 20 clients: {acc_20_iid - acc_5_iid:+.2f}%")

    print(f"\n4. Impact of Client Count (Non-IID):")
    acc_5_non_iid = results['fl_non_iid_5_clients']['final_accuracy'] * 100
    acc_10_non_iid = results['fl_non_iid_10_clients']['final_accuracy'] * 100
    acc_20_non_iid = results['fl_non_iid_20_clients']['final_accuracy'] * 100
    print(f"   - 5 clients:  {acc_5_non_iid:.2f}%")
    print(f"   - 10 clients: {acc_10_non_iid:.2f}%")
    print(f"   - 20 clients: {acc_20_non_iid:.2f}%")
    print(f"   - Change from 5 to 20 clients: {acc_20_non_iid - acc_5_non_iid:+.2f}%")

    print(f"\n5. Convergence Speed Comparison (Rounds to reach 95% of final accuracy):")
    print(f"   IID:")
    for clients in client_counts:
        conv_round = results[f'fl_iid_{clients}_clients']['convergence_round']
        print(f"   - {clients} clients:  {conv_round} rounds")
    print(f"   Non-IID:")
    for clients in client_counts:
        conv_round = results[f'fl_non_iid_{clients}_clients']['convergence_round']
        print(f"   - {clients} clients:  {conv_round} rounds")

    # Compare IID vs Non-IID convergence
    conv_iid_10 = results['fl_iid_10_clients']['convergence_round']
    conv_non_iid_10 = results['fl_non_iid_10_clients']['convergence_round']
    print(f"   Impact of Non-IID on convergence (10 clients): {conv_non_iid_10 - conv_iid_10:+d} rounds")

    print("\n" + "=" * 80)
    print(f"Experiments completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_all_experiments()
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("\nGenerated files:")
    print("\n[Models]")
    print("  - models/centralised_model.pth")
    print("  - models/fl_5_clients_iid.pth")
    print("  - models/fl_10_clients_iid.pth")
    print("  - models/fl_20_clients_iid.pth")
    print("  - models/fl_5_clients_non_iid.pth")
    print("  - models/fl_10_clients_non_iid.pth")
    print("  - models/fl_20_clients_non_iid.pth")
    print("\n[Plots]")
    print("  - plots/centralised_training.png")
    print("  - plots/experiment1_client_counts.png")
    print("  - plots/experiment2_iid_vs_non_iid.png")
    print("  - plots/experiment3_convergence_comparison.png")
    print("  - plots/comprehensive_comparison.png")
    print("\n[Results]")
    print("  - results/experiment_results.json")
    print("  - results/system_info.json")
    print("  - results/config.json")
    print("="*80)

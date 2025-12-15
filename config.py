"""
Unified configuration for all experiments.
Ensures fair comparison across centralised and federated learning methods.
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Dataset parameters
DATASET_NAME = 'Fashion-MNIST'
NUM_CLASSES = 10
INPUT_SIZE = 784  # 28x28 flattened
TRAIN_SAMPLES = 60000
TEST_SAMPLES = 10000

# Model architecture parameters
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64

# Training hyperparameters (MUST BE SAME FOR ALL METHODS)
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
OPTIMIZER_TYPE = 'Adam'  # Options: 'Adam', 'SGD'
LOSS_FUNCTION = 'CrossEntropyLoss'

# Centralised learning parameters
CENTRALISED_EPOCHS = 15  # Adjusted for fair comparison with FL (20 rounds × 3 local epochs)

# Federated learning parameters
FL_NUM_ROUNDS = 20
FL_LOCAL_EPOCHS = 3
FL_CLIENT_COUNTS = [5, 10, 20]
FL_DATA_DISTRIBUTIONS = ['iid', 'non_iid']
FL_NON_IID_CLASSES_PER_CLIENT = 2

# Hardware parameters
NUM_WORKERS = 8  # For data loading

# Output directories
PLOTS_DIR = 'plots'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

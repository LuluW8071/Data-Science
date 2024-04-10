import torch

# Setup a device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Training Hyperparams
INPUT_SIZE = 784
HIDDEN_SIZE = 128
NUM_CLASSES = 10
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Dataset Params
DATA_DIR = "dataset/"

# Compute Related Params
ACCELERATOR = device
DEVICES = 1             # Multi-Gpu = 2
NUM_WORKERS = 2      
PRECISION = 16
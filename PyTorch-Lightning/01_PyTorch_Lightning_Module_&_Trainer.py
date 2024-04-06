""" PyTorch Lightning Module and Trainer """

import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

# Import PyTorch Lightning
import pytorch_lightning as pl

# NOTE: =======================================
# A LightningModule organizes your PyTorch code into 6 sections:
# - Initialization                      (__init__ and setup()).
# - Train Loop                          (training_step())
# - Validation Loop                     (validation_step())
# - Test Loop                           (test_step())
# - Prediction Loop                     (predict_step())
# - Optimizers and LR Schedulers        (configure_optimizers())

# When you convert to use Lightning, the code IS NOT abstracted - just organized.
# All the other code that’s not in the LightningModule has been automated for you by the Trainer.
# =====================================


# Build a linear neuralnet using `pl.LightningModule`
class neuralnet(pl.LightningModule):
    # Initialize the model here
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass function representing how the data flows through the model.
        This function defines the architecture of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            - Pass input through the first fully connected layer (FC1)
              followed by a ReLU activation function
            - Pass the output through the second fully connected layer (FC2)
              followed by another ReLU activation function
            - Pass the output through the third fully connected layer (FC3)
              without an activation function, as this might be the output layer

            torch.Tensor: Output tensor representing the predictions of the model.
        """

        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        return x     # Return the output tensor representing predictions

    def _common_step(self, batch, batch_idx):
        """
        Common step function used by training_step, validation_step, and test_step.
        This function preprocesses the batch, computes predictions, calculates loss,
        and returns necessary values.

        Args:
            batch (tuple): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            tuple: A tuple containing loss, predicted outputs, and ground truth labels.
        """
        X, y = batch                    # Extract input data and labels from the batch
        X = X.reshape(X.shape[0], -1)   # Reshape the input data
        y_pred = self.forward(X)        # Compute predictions using the model
        loss = self.loss_fn(y_pred, y)  # Calculate the loss
        # Return loss, predicted outputs, and ground truth labels
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        """
        Training step function.
        This function is called for each batch during the training loop.
        It calculates the loss, logs it, and returns the loss value.

        Args:
            batch (tuple): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            float: Loss value for the current batch.
        """
        loss, y_pred, y = self._common_step(
            batch, batch_idx)  # Call _common_step to get loss
        self.log("train_loss", loss)    # Log training loss
        return loss                     # Return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step function.
        This function is called for each batch during the validation loop.
        It calculates the loss, logs it, and returns the loss value.

        Args:
            batch (tuple): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            float: Loss value for the current batch.
        """
        loss, y_pred, y = self._common_step(
            batch, batch_idx)  # Call _common_step to get loss
        self.log("val_loss", loss)  # Log validation loss
        return loss                 # Return loss

    def test_step(self, batch, batch_idx):
        """
        Test step function.
        This function is called for each batch during the testing loop.
        It calculates the loss, logs it, and returns the loss value.

        Args:
            batch (tuple): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            float: Loss value for the current batch.
        """
        loss, y_pred, y = self._common_step(
            batch, batch_idx)  # Call _common_step to get loss
        self.log("test_loss", loss)  # Log test loss
        return loss  # Return loss

    def predict_step(self, batch, batch_idx):
        """
        Prediction step function.
        This function is used during inference/prediction.
        It computes predictions for the input batch and returns them.

        Args:
            batch (tuple): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Predicted outputs for the input batch.
        """
        X = batch                         # Extract input data and labels from the batch
        X = X.reshape(X.shape[0], -1)        # Flatten the input data if needed
        # Compute predictions using the model
        y_pred = self.forward(X)
        # Convert predicted probabilities to class labels
        preds = torch.argmax(y_pred, dim=1)
        return preds                         # Return predicted outputs

    def configure_optimizers(self):
        """
        Configure optimizer function.
        This function specifies the optimizer to be used for training.

        Returns:
            torch.optim.Optimizer: Optimizer instance (e.g., Adam optimizer) configured with model parameters.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)  # Return Adam optimizer


# Setup a device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Load Data
dataset = datasets.MNIST(root="datasets/",
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

# Random split the datasets into train and validation set
train_data, val_data = random_split(dataset, [50000, 10000])

# Load test data
test_data = datasets.MNIST(root="dataset/",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

# Hyperparams
input_size = 784
hidden_size = 128
num_classes = 10
batch_size = 64

# Create the dataloader
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=False)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)

# Initialize the model
model = neuralnet(input_size, hidden_size, num_classes).to(device)

# Create a Trainer instance for managing the training process.
# Set the accelerator to "cuda" to utilize GPU for computation.
# Specify the number of devices (GPUs) to be used for training as 1 or [1,2] for multi-gpu train.
# Set the minimum number of epochs to 1 and the maximum number of epochs to 5.
# Use mixed-precision training with 16-bit floating-point numbers for faster training.
trainer = pl.Trainer(accelerator="cuda",
                     devices=1,
                     min_epochs=1,
                     max_epochs=3,
                     precision=16)

# Fit the model to the training data using the Trainer's fit method.
# Pass the model instance, training dataloader, and validation dataloader.
trainer.fit(model, train_dataloader, val_dataloader)

# Validate the trained model using the validation dataloader.
# This step evaluates the model's performance on the validation set.
trainer.validate(model, val_dataloader)

# Test the trained model using the test dataloader.
# This step evaluates the model's performance on the test set.
trainer.test(model, test_dataloader)


# ================================================================
# DEMO RESULTS:

# INFO:pytorch_lightning.utilities.rank_zero:Using 16bit Automatic Mixed Precision (AMP)
# INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
# INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
# INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
# INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# INFO:pytorch_lightning.callbacks.model_summary:
#   | Name    | Type             | Params
# ---------------------------------------------
# 0 | fc1     | Linear           | 100 K
# 1 | fc2     | Linear           | 16.5 K
# 2 | fc3     | Linear           | 1.3 K
# 3 | relu    | ReLU             | 0
# 4 | loss_fn | CrossEntropyLoss | 0
# ---------------------------------------------
# 118 K     Trainable params
# 0         Non-trainable params
# 118 K     Total params
# 0.473     Total estimated model params size (MB)
# cuda
# Epoch 4: 100% |██████████████████████████████████████████████████████████████████| 782/782 [00:09<00:00, 78.33it/s, v_num=2]
# INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=5` reached.
# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# Validation DataLoader 0: 100%
#  157/157 [00:01<00:00, 137.48it/s]
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃      Validate metric      ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         val_loss          │    0.11773595213890076    │
# └───────────────────────────┴───────────────────────────┘
# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# Testing DataLoader 0: 100%
#  157/157 [00:01<00:00, 137.97it/s]
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         test_loss         │    0.10124694555997849    │
# └───────────────────────────┴───────────────────────────┘
# [{'test_loss': 0.10124694555997849}]
# ================================================================

""" PyTorch Lightning Metrics """

import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

# Import torchmetrics library: https://lightning.ai/docs/torchmetrics/stable/
# NOTE: =======================================
# TorchMetrics is a collection of 100+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:

# - A standardized interface to increase reproducibility
# - Reduces Boilerplate
# - Distributed-training compatible
# - Rigorously tested
# - Automatic accumulation over batches
# - Automatic synchronization between multiple devices

# You can use TorchMetrics in any PyTorch model, or within PyTorch Lightning to enjoy the following additional benefits:
# Your data will always be placed on the same device as your metrics
# You can log Metric objects directly in Lightning to reduce even more boilerplate
# ==============================================
import torchmetrics
from torchmetrics import Metric

# ==============================================
# NOTE: This is not super important just to see how you can customize the metrics
# Can be helpful when training models on multiple gpus to aggregate the calcualted metrics 
# Custom Metrics can also be defined using class method
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()  
        # Add states for tracking total and correct predictions
        self.add_state("total", 
                       default=torch.tensor(0),  # Initial value: 0
                       dist_reduce_fx="sum")     # Reduction: "sum" for distributed training
        self.add_state("correct", 
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
    
    # Update method for updating metric with new predictions and targets
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)          # Get predicted classes
        assert preds.shape == target.shape          # Assert same shape
        self.correct += torch.sum(preds == target)  # Update correct predictions
        self.total += target.numel(0)               # Update total count

    # Compute method to calculate accuracy
    def compute(self):
        return self.correct.float() / self.total.float()  # Calculate accuracy ratio
# ==============================================


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

        # NOTE: ============================================
        # Previously in `00_PyTorch_Basic_Implementation` we created accuracy function
        # to calculate the accuracy. That is extremely time consuming and often prone to errors.
        # We can just use `torchmetrics` api calls for much more metrics instead of 
        # writing individual logic for it.
        # ==================================================
        # Create accuracy and f1_score instance using torchmetrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) 
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes) 

        # Instance of custom metric MyAccuracy
        self.my_accuracy = MyAccuracy()
    def forward(self, x):
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        return x                        # Return the output tensor representing predictions

    def _common_step(self, batch, batch_idx):
        X, y = batch                    # Extract input data and labels from the batch
        X = X.reshape(X.shape[0], -1)   # Reshape the input data
        y_pred = self.forward(X)        # Compute predictions using the model
        loss = self.loss_fn(y_pred, y)  # Calculate the loss
        return loss, y_pred, y          # Return loss, predicted outputs, and ground truth labels

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)   # Call _common_step to get loss
        accuracy = self.accuracy(y_pred, y)                     # Calculate train_accuracy
        # accuracy = self.my_accuracy(y_pred, y) -------------- # Calcualte accuracy using custom metrics MyAccuracy
        f1_score = self.f1_score(y_pred, y)                     # Calculate train_f1_score   
        self.log_dict({"loss": loss,                            # Log train_loss,
                       "accuracy": accuracy,                    # train_accuracy,
                       "f1_score": f1_score},                   # train_f1_score metrics in the form of dict
                       on_step=True, on_epoch=False,            # Log the metrics on every global step but not on end of epochs 
                       prog_bar=True,                           # Show the metrics on progress bar
                       logger=True)                             # Log parameter
        return loss                                             # Return the loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)  # Call _common_step to get loss
        self.log("val_loss", loss)                             # Log validation loss
        return loss                                            # Return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)   # Call _common_step to get loss
        self.log("test_loss", loss)                             # Log test loss
        return loss                                             # Return loss

    def predict_step(self, batch, batch_idx):
        X = batch                                   # Extract input data and labels from the batch
        X = X.reshape(X.shape[0], -1)               # Flatten the input data if needed
        y_pred = self.forward(X)                    # Compute predictions using the model
        preds = torch.argmax(y_pred, dim=1)         # Convert predicted probabilities to class labels
        return preds                                # Return predicted outputs

    def configure_optimizers(self):
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
batch_size = 128

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
trainer = pl.Trainer(accelerator="cuda",
                     devices=1,
                     min_epochs=1,
                     max_epochs=3,
                     precision=16)

# Fit the model to the training data using the Trainer's fit method.
trainer.fit(model, train_dataloader, val_dataloader)
trainer.validate(model, val_dataloader)
trainer.test(model, test_dataloader)


# ================================================================
# INFO:pytorch_lightning.utilities.rank_zero:Using 16bit Automatic Mixed Precision (AMP)
# INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
# INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
# INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
# INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# INFO:pytorch_lightning.callbacks.model_summary:
#   | Name        | Type               | Params
# ---------------------------------------------------
# 0 | fc1         | Linear             | 100 K 
# 1 | fc2         | Linear             | 16.5 K
# 2 | fc3         | Linear             | 1.3 K 
# 3 | relu        | ReLU               | 0     
# 4 | loss_fn     | CrossEntropyLoss   | 0     
# 5 | accuracy    | MulticlassAccuracy | 0     
# 6 | f1_score    | MulticlassF1Score  | 0     
# 7 | my_accuracy | MyAccuracy         | 0     
# ---------------------------------------------------
# 118 K     Trainable params
# 0         Non-trainable params
# 118 K     Total params
# 0.473     Total estimated model params size (MB)

# cuda

# Epoch 2: 100%|██████████████████████████████████████████████████████████████████| 782/782 [00:11<00:00, 67.16it/s, v_num=4, loss=0.0175, accuracy=1.000, f1_score=1.000]

# INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=3` reached.
# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

# Validation DataLoader 0: 100%|██████████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 113.94it/s]

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃      Validate metric      ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         val_loss          │    0.11501604318618774    │
# └───────────────────────────┴───────────────────────────┘

# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

# Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 93.88it/s]

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         test_loss         │    0.09931295365095139    │
# └───────────────────────────┴───────────────────────────┘

# [{'test_loss': 0.09931295365095139}]
# ================================================================

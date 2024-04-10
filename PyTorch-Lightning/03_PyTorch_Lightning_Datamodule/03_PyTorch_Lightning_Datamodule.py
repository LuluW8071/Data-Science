""" PyTorch Lightning Data Module """
# Now we begin writing the code using the actual PyTorch Lightning approach.

import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
import torchmetrics

# =================================================================================
# NOTE:
# What is a Data Module?
# The LightningDataModule is a convenient way to manage data in PyTorch Lightning. 
# It encapsulates training, validation, testing, and prediction dataloaders, as well as any necessary steps for data processing, downloads, and transformations.
# By using a LightningDataModule, you can easily develop dataset-agnostic models, hot-swap different datasets, and share data splits and transformations across projects.
# =================================================================================

# Create a MNISTDataModule
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir            # Directory containing MNIST data
        self.batch_size = batch_size        # Batch size for DataLoader
        self.num_workers = num_workers      # Number of workers (No of CPU allocation) for DataLoader

    def prepare_data(self):
        """ 
        Prepare MNIST dataset for training and testing 
        - Downloading and saving data with multiple processes (distributed settings) will result in corrupted data. 
        - Lightning ensures the prepare_data() is called only within a single process on CPU, so you can safely add your downloading logic within.
        """
        datasets.MNIST(self.data_dir, train=True, download=True)    # Download MNIST Train Dataset
        datasets.MNIST(self.data_dir, train=False, download=True)   # Download MNIST Test Dataset

    def setup(self, stage):
        """ 
        Download and Split the entire MNIST dataset into train, validation, and test sets 
        - In case of multi-node training, the execution of this hook depends upon `prepare_data_per_node`. 
        - `setup()` is called after prepare_data and there is a barrier in between which ensures that all the processes proceed to setup once the data is prepared and available for use.
        
        ---`setup` is called from every process across all the nodes. Setting state here is recommended.---
        """
        entire_dataset = datasets.MNIST(root=self.data_dir,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=False)
        self.train_data, self.val_data = random_split(entire_dataset, [50000, 10000])
        self.test_data = datasets.MNIST(root=self.data_dir,
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=False)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)


# Build a linear neuralnet
class neuralnet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) 
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes) 

    def forward(self, x):
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        return x

    def _common_step(self, batch, batch_idx):
        X, y = batch                    # Extract input data and labels from the batch
        X = X.reshape(X.shape[0], -1)   # Reshape the input data
        y_pred = self.forward(X)        # Compute predictions using the model
        loss = self.loss_fn(y_pred, y)  # Calculate the loss
        return loss, y_pred, y          # Return loss, predicted outputs, and ground truth labels

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)   # Call _common_step to get loss
        accuracy = self.accuracy(y_pred, y)                     # Calculate train_accuracy
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
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Setup a device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# Hyperparams
input_size = 784
hidden_size = 128
num_classes = 10
batch_size = 128

# Initialize the MNISTDataModule Dataloader 
dataloader = MNISTDataModule(data_dir="dataset/", 
                             batch_size=batch_size, 
                             num_workers=2)

# Initialize the model
model = neuralnet(input_size, hidden_size, num_classes).to(device)

# Create a Trainer instance for managing the training process.
trainer = pl.Trainer(accelerator="cuda",
                     devices=1,
                     min_epochs=1,
                     max_epochs=5,
                     precision=16)

# Fit the model to the training data using the Trainer's fit method.
# NOTE: Changed previous individual dataloder to single dataloader instance
# This will automatically train, val, test dataloaders
trainer.fit(model, dataloader)
trainer.validate(model, dataloader)
trainer.test(model, dataloader)


# DEMO RESULTS:
# ================================================================
# INFO:pytorch_lightning.utilities.rank_zero:Using 16bit Automatic Mixed Precision (AMP)
# INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
# INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
# INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
# INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs

# cuda

# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# INFO:pytorch_lightning.callbacks.model_summary:
#   | Name     | Type               | Params
# ------------------------------------------------
# 0 | fc1      | Linear             | 100 K 
# 1 | fc2      | Linear             | 16.5 K
# 2 | fc3      | Linear             | 1.3 K 
# 3 | relu     | ReLU               | 0     
# 4 | loss_fn  | CrossEntropyLoss   | 0     
# 5 | accuracy | MulticlassAccuracy | 0     
# 6 | f1_score | MulticlassF1Score  | 0     
# ------------------------------------------------
# 118 K     Trainable params
# 0         Non-trainable params
# 118 K     Total params
# 0.473     Total estimated model params size (MB)

# Epoch 4: 100%|██████████████████████████████████████████████████████████████████| 391/391 [00:10<00:00, 36.70it/s, v_num=1, loss=0.0298, accuracy=1.000, f1_score=1.000]

# INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=5` reached.
# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

# Validation DataLoader 0: 100%|██████████████████████████████████████████████████████████████████| 79/79 [00:01<00:00, 62.17it/s]

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃      Validate metric      ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         val_loss          │    0.06989479064941406    │
# └───────────────────────────┴───────────────────────────┘

# INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

# Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████████████| 79/79 [00:01<00:00, 60.64it/s]

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         test_loss         │    0.09798655658960342    │
# └───────────────────────────┴───────────────────────────┘

# [{'test_loss': 0.09798655658960342}]
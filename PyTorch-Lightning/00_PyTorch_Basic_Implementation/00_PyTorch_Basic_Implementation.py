""" Basic PyTorch Workflow for training MNIST Dataset """

import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy
from tqdm.auto import tqdm

# Build a linear neuralnet
class neuralnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        return x


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
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# Create the dataloader
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)

# Initialize the model
model = neuralnet(input_size, hidden_size, num_classes).to(device)

# Define the loss function, optimizer and accuracy
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)

accuracy = Accuracy(task='multiclass',
                    num_classes=num_classes).to(device)

# Train the model
for epoch in tqdm(range(num_epochs)):
    train_loss = 0
    # Training loop
    for batch, (X, y) in enumerate(train_dataloader):
        model.train()
        # Send data and targets to selected device
        X, y = X.to(device), y.to(device)
        X = X.reshape(X.shape[0], -1)        # Reshape the data
        y_pred = model(X)                    # Train model
        loss = loss_fn(y_pred, y)            # Calculate loss
        train_loss += loss                   # Accumulate loss
        optimizer.zero_grad()                # Optimizer zero gradient
        loss.backward()                      # Backward loss
        optimizer.step()                     # Optimizing descent or Adam step

    train_loss /= len(train_dataloader)

    # Validation and Test
    val_loss, val_acc = 0, 0
    test_loss, test_acc = 0, 0
    model.eval()
    # Validation loop
    with torch.inference_mode():
        for batch, (X_val, y_val) in enumerate(val_dataloader):
            X_val, y_val = X_val.to(device), y_val.to(device)
            X_val = X_val.reshape(X_val.shape[0], -1)
            val_pred = model(X_val)
            val_loss += loss_fn(val_pred, y_val)
            val_acc += accuracy(y_val, val_pred.argmax(dim=1)) * 100      # Calculate accuracy

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

    # Testing loop
        for batch, (X_test, y_test) in enumerate(test_dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test = X_test.reshape(X_test.shape[0], -1)
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy(y_test, test_pred.argmax(dim=1)) * 100

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    # Print results
    print(f'\nTrain loss: {train_loss:.4f} | Validation loss: {val_loss:.4f} --- Validation acc: {val_acc:.2f}% | Test loss: {test_loss:.4f} --- Test acc: {test_acc:.2f}%')


# DEMO RESULTS:
#   0%|                                                                                                                    | 0/3 [00:00<?, ?it/s]
#   Train loss: 0.3510 | Validation loss: 0.1842 --- Validation acc: 94.67% | Test loss: 0.1782 --- Test acc: 94.52%
#  33%|████████████████████████████████████                                                                        | 1/3 [00:36<01:13, 36.63s/it]
#  Train loss: 0.1454 | Validation loss: 0.1305 --- Validation acc: 95.98% | Test loss: 0.1286 --- Test acc: 96.18%
#  67%|████████████████████████████████████████████████████████████████████████                                    | 2/3 [01:02<00:30, 30.19s/it]
#  Train loss: 0.0994 | Validation loss: 0.1025 --- Validation acc: 96.75% | Test loss: 0.0959 --- Test acc: 97.07%
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [01:26<00:00, 28.68s/it]
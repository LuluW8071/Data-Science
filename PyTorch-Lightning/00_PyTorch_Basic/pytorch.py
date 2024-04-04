""" Basic PyTorch Workflow for training MNIST Dataset """

import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
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
num_epochs = 5

# Create the dataloader
train_dataloader = DataLoader(dataset = train_data,
                              batch_size = batch_size,
                              shuffle = True)

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

# Train the model
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        # Send data to targeted device
        data, targets = data.to(device), targets.to(device)

        # Reshape the data
        data = data.reshape(data.shape[0], -1)

        # Forward pass
        output = model(data)
        loss = loss_fn(output, targets)

        # Backward loss
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent
        optimizer.step()

# Create accuracy function on train and test 
def accuracy(loader, model):
    num_correct = 0 
    num_samples = 0 
    model.eval()
    with torch.inference_mode():
        for X,y in tqdm(loader):
            X, y = X.to(device), y.to(device)
            X = X.reshape(X.shape[0], -1)
            y_pred = model(X)
            _, preds = y_pred.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.shape[0]

    model.train()
    return num_correct / num_samples

# Check accuracy on training & test to see how good our model
model.to(device)
print(f"Acc on train_set: {accuracy(train_dataloader, model)*100:.2f}")
print(f"Acc on val_set: {accuracy(val_dataloader, model)*100:.2f}")
print(f"Acc on test set: {accuracy(test_dataloader, model)*100:.2f}")

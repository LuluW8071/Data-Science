import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model, dataloader, loss_fn, optimizer, device):
    """
    Perform a single training step.

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        loss_fn (torch.nn.Module): The loss function to use for training.
        optimizer (torch.optim): The optimizer to use for training.
        device (torch.device): The device to use for training.

    Returns:
        Tuple[float, float]: A tuple containing the average training loss and accuracy.

    """
    # Put model on train mode
    model.train()

    # Setup train loss and accuracy values
    train_loss, train_acc = 0, 0

    # Loop through dataloader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)   # Send data to target device

        y_pred = model(X)                   # Forward pass
        loss = loss_fn(y_pred, y)           # Calculate loss
        train_loss += loss.item()           # Accumulate loss
        optimizer.zero_grad()               # Optimizer zero grad
        loss.backward()                     # Loss backward
        optimizer.step()                    # Optimizer step

        # Calculate and ammulate accuracy metrics across all batches 
        y_pred_class = torch.argmax(torch.softmax(y_pred,
                                                dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    # Adjust metrics to get avg. loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device):
    """
    Perform a single testing step.

    Args:
        model (torch.nn.Module): The neural network model to test.
        dataloader (torch.utils.data.DataLoader): The testing data loader.
        loss_fn (torch.nn.Module): The loss function to use for testing.
        device (torch.device): The device to use for testing.

    Returns:
        Tuple[float, float]: A tuple containing the average testing loss and accuracy.

    """
    # Put model on evaluation mode
    model.eval()

    # Setup test loss and accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through dataloader data batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)       # Send data to target device

            test_pred_logits = model(X)             # Forward pass
            loss = loss_fn(test_pred_logits, y)     # Calculate loss
            test_loss += loss.item()                # Accumulate loss

            # Calculate and ammulate accuracy metrics across all batches 
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)
    
    # Adjust metrics to get avg. loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model, train_dataloader, test_dataloader, 
          loss_fn, optimizer, epochs, device):
    """
    Train a neural network model for a given number of epochs.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_dataloader (torch.utils.data.DataLoader): The training data loader.
        test_dataloader (torch.utils.data.DataLoader): The testing data loader.
        loss_fn (torch.nn.Module): The loss function to use for training and testing.
        optimizer (torch.optim): The optimizer to use for training.
        epochs (int): The number of epochs to train for.
        device (torch.device): The device to use for training and testing.

    Returns:
        Dict[str, List[float]]: A dictionary containing the training and testing loss and accuracy for each epoch.

    """
    # Create empty results dictionary 
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Loop through train_steps and test_steps for no of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,
                                           train_dataloader,
                                           loss_fn,
                                           optimizer,
                                           device)
        test_loss, test_acc = test_step(model,
                                        test_dataloader,
                                        loss_fn,
                                        device)

        # Print out metrics
        print(f"\nEpoch: {epoch+1} | Train loss: {train_loss:.4f} - Train acc: {(train_acc*100):.2f}% -- Test_loss: {test_loss:.4f} -- Test_acc: {(test_acc*100):.2f}%")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
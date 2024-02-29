
import os 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers):
    """
    Creates train and test dataloaders.

    Args:
        train_dir (str): path to the training data directory
        test_dir (str): path to the testing data directory
        transform (torchvision.transforms): transformation to apply to the images
        batch_size (int): batch size to use for training and testing
        num_workers (int): number of workers to use for data loading

    Returns:
        tuple: containing:
            train_dataloader (torch.utils.data.DataLoader): dataloader for the training data
            test_dataloader (torch.utils.data.DataLoader): dataloader for the testing data
            class_names (list): list of class names

    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get the class names
    class_names = train_data.classes

    # Turn images into dataloaders
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names

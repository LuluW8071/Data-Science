import pytorch_lightning as pl

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


class MNISTFoodDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size, num_workers):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage):
        # Define transformation for image
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor()
        ])

        train_dataset = ImageFolder(root=self.train_dir, transform=data_transform)
        self.test_data = ImageFolder(root=self.test_dir, transform=data_transform)

        # Random split for train and val dataset
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        self.train_data, self.val_data = random_split(train_dataset, [train_size, test_size])
        self.num_classes = len(train_dataset.classes)

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
    
    def get_num_classes(self):
        return self.num_classes

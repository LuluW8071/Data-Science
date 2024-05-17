import pytorch_lightning as pl

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import random_split, DataLoader


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(root=self.data_dir,
                                        train=True,
                                        transform=transforms.Compose([
                                                transforms.RandomVerticalFlip(),        # Actually not needed for MNIST datasets as it will completely break the learning od model
                                                transforms.RandomHorizontalFlip(),      # VerticalFlip() and HorizontalFlip() is just applied for this tensorboard workflow                                         
                                                transforms.ToTensor(),           
                                            ]),
                                        download=False)
        self.test_data = datasets.MNIST(root=self.data_dir,
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=False)
        self.train_data, self.val_data = random_split(entire_dataset, [50000, 10000])

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

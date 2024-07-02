import torch
import pytorch_lightning as pl 

from model import neuralnet
from dataset import MNISTFoodDataModule
import argparse

def main(args):
    # Setting up device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = MNISTFoodDataModule(train_dir=args.train_dir,
                                     test_dir=args.test_dir, 
                                     batch_size=args.batch_size, 
                                     num_workers=args.data_workers)
    
    # Call setup to initialize datasets
    dataloader.setup('fit')  
    num_classes = dataloader.get_num_classes()

    # Initialize the model
    model = neuralnet(num_classes=num_classes).to(device)

    # Create a Trainer instance for managing the training process.
    trainer = pl.Trainer(accelerator=device,
                         devices=args.gpus,
                         min_epochs=1,
                         max_epochs=args.epochs,
                         precision=args.precision)

    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)
    trainer.test(model, dataloader)


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Train")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=0, type=int,
                        help='n data loading workers, default 0 = main process only')

    # Train and Test Directory Params
    parser.add_argument('--train_dir', default=None, required=True, type=str,
                        help='Folder path to load training data')
    parser.add_argument('--test_dir', default=None, required=True, type=str,
                        help='Folder path to load testing data')
    
    # General Train Hyperparameters
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('--precision', default=16, type=int, help='precision')
    
    args = parser.parse_args()
    main(args)

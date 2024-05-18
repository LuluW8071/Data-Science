import torch
from torch import nn

import pytorch_lightning as pl
import torchmetrics
import torchvision
import config


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
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

    def _common_step(self, batch, batch_idx):
        X, y = batch 
        X = X.reshape(X.shape[0], -1) 
        y_pred = self.forward(X) 
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y 

    def training_step(self, batch, batch_idx):
        x, y = batch    # Pass inputs and ouputs to batch
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y) 
        f1_score = self.f1_score(y_pred, y)  
        self.log_dict({"loss": loss,  
                       "accuracy": accuracy,  
                       "f1_score": f1_score}, 
                       on_step=True, on_epoch=False, 
                       prog_bar=True,
                       logger=True)

        if batch_idx % 100 == 0:
            x = x[:8]       # How many images to log on tensorboard image view at each 100 steps batch
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("MNIST_images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx) 
        self.log("val_loss", loss, prog_bar=True)       # Log valdiation loss in prog bar
        return loss 

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss) 
        return loss 

    def predict_step(self, batch, batch_idx):
        X = batch  
        X = X.reshape(X.shape[0], -1) 
        y_pred = self.forward(X)
        preds = torch.argmax(y_pred, dim=1) 
        return preds 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)

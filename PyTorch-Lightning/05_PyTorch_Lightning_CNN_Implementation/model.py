import torch
from torch import nn 
from torchvision.models import models
import pytorch_lightning as pl 
import torchmetrics

import argparse

class neuralnet(pl.LightningModule):
    def __init__(self, num_classes):
        super(neuralnet, self).__init__()
        self.weights = models.efficientnet.b7(pretrained=True)          # Load pre-trained weights
        self.model = models.efficientnet.EfficientNet.from_pretrained('efficientnet-b7')

        # Modify the classifier layer
        # Get the number of input features to the classifier
        num_ftrs = self.model._fc.in_features  
        self.model._fc = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),
                                             torch.nn.Linear(in_features=num_ftrs, 
                                                             out_features=num_classes, 
                                                             bias=True))
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) 
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes) 
        
    def forward(self, x):
        return self.model(x)
    
    def _common_step(self, batch, batch_idx):
        X, y = batch                       
        y_pred = self.forward(X)       
        loss = self.loss_fn(y_pred, y) 
        return loss, y_pred, y
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)   # Call _common_step to get loss
        accuracy = self.accuracy(y_pred, y)  
        f1_score = self.f1_score(y_pred, y)
        self.log_dict({"loss": loss, 
                        "accuracy": accuracy, 
                        "f1_score": f1_score}, 
                        on_step=True, on_epoch=False, 
                        prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)  # Call _common_step to get loss
        accuracy = self.accuracy(y_pred, y)  
        f1_score = self.f1_score(y_pred, y)
        self.log_dict({"loss": loss, 
                        "accuracy": accuracy, 
                        "f1_score": f1_score}, 
                        on_step=True, on_epoch=False, 
                        prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)  # Call _common_step to get loss
        accuracy = self.accuracy(y_pred, y)  
        f1_score = self.f1_score(y_pred, y)
        self.log_dict({"loss": loss, 
                        "accuracy": accuracy, 
                        "f1_score": f1_score}, 
                        on_step=True, on_epoch=False, 
                        prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        X = batch
        y_pred = self.forward(X)
        preds = torch.argmax(y_pred, dim=1) 
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    

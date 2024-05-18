import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

""" TensorBoard """
# TensorBoard is an interactive visualization toolkit for machine learning experiments. Essentially it is a web-hosted app that lets us understand our model’s training run and graphs.

# Default TensorBoard Logging: Logging per batch
# Lightning gives us the provision to return logs after every forward pass of a batch, which allows TensorBoard to automatically make plots. 
# Referemce Link: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html

""" Other Loggers similar to TensorBoard """
# 1. Comet Logger,
# 2. Neptune Logger,
# 3. MLflow Logger and many more

import config
from model import neuralnet
from dataset import MNISTDataModule
from callback import (printCallback)

if __name__ == "__main__":
    # Logging with TensorBoard
    logger = TensorBoardLogger("tb_logs", name="MNIST_model")


    dataloader = MNISTDataModule(data_dir=config.DATA_DIR,
                                 batch_size=config.BATCH_SIZE,
                                 num_workers=config.NUM_WORKERS)

    model = neuralnet(config.INPUT_SIZE,
                      config.HIDDEN_SIZE,
                      config.NUM_CLASSES).to(config.ACCELERATOR)

    # Create a checkpoint callback
    # Save the model periodically by monitoring a quantity
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",                    # Quantity to monitor | By default: None(saves last chekpoint)
                                          dirpath="./saved_checkpoint/",         # Directory to save the model file.
                                          filename="MSNIT-model-{epoch:02d}-{val_loss:.2f}")   # Checkpoint filename

    # Create a Trainer instance with callbacks and logger
    trainer = pl.Trainer(logger = logger,     # This will automatically log all the metrics we have defined 
                         accelerator=config.ACCELERATOR,
                         devices=config.DEVICES,
                         min_epochs=1,
                         max_epochs=config.NUM_EPOCHS,
                         precision=config.PRECISION,
                         callbacks=[printCallback(),                     # For Custom Callback hook in callback.py
                                    EarlyStopping(monitor="val_loss"),   # Monitor a metric and stop training when it stops improving
                                    checkpoint_callback])                # Pass defined checkpoint callback

    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)
    trainer.test(model, dataloader)

# DEMO RESULTS:
# ======================================================================================

# /home/codespace/.python/current/lib/python3.10/site-packages/lightning_fabric/connector.py:563: `precision=16` is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
# /home/codespace/.python/current/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:556: You passed `Trainer(accelerator='cpu', precision='16-mixed')` but AMP with fp16 is not supported on CPU. Using `precision='bf16-mixed'` instead.
# Using bfloat16 Automatic Mixed Precision (AMP)
# GPU available: False, used: False
# TPU available: False, using: 0 TPU cores
# IPU available: False, using: 0 IPUs
# HPU available: False, using: 0 HPUs

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
# Start train!                                                                                                                                                                                   
# Epoch 4: 100%|██████████████████████████████████████████████████████████████████████████| 391/391 [00:19<00:00, 20.26it/s, v_num=2, loss=0.374, accuracy=0.863, f1_score=0.863, val_loss=0.267]`Trainer.fit` stopped: `max_epochs=5` reached.                                                                                                                                                 
# End train!
# Epoch 4: 100%|██████████████████████████████████████████████████████████████████████████| 391/391 [00:19<00:00, 20.25it/s, v_num=2, loss=0.374, accuracy=0.863, f1_score=0.863, val_loss=0.267]
# Validation DataLoader 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 79/79 [00:02<00:00, 38.54it/s]
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#      Validate metric           DataLoader 0
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#         val_loss            0.22596271336078644
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Testing DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 79/79 [00:01<00:00, 49.43it/s]
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#         test_loss           0.23988471925258636
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import config
from model import neuralnet
from dataset import MNISTDataModule
from callback import (printCallback)

if __name__ == "__main__":
    dataloader = MNISTDataModule(data_dir=config.DATA_DIR,
                                 batch_size=config.BATCH_SIZE,
                                 num_workers=config.NUM_WORKERS)

    model = neuralnet(config.INPUT_SIZE,
                      config.HIDDEN_SIZE,
                      config.NUM_CLASSES).to(config.ACCELERATOR)

    # Create a checkpoint callback
    # Save the model periodically by monitoring a quantity
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",                    # Quantity to monitor | By default: None(saves last chekpoint)
                                          dirpath="./saved_checkpoint/",           # Directory to save the model file.
                                          filename="MSNIT-model-{epoch:02d}-{val_loss:.2f}")   # Checkpoint filename

    # Create a Trainer instance with callbacks
    trainer = pl.Trainer(accelerator=config.ACCELERATOR,
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

# /usr/local/lib/python3.10/dist-packages/lightning_fabric/connector.py:563: `precision=16` is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
# Using 16bit Automatic Mixed Precision (AMP)
# GPU available: False, used: False
# TPU available: False, using: 0 TPU cores
# IPU available: False, using: 0 IPUs
# HPU available: False, using: 0 HPUs
# /home/codespace/.python/current/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
# Missing logger folder: /workspaces/dev/lightning_logs

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
# Epoch 2: 100%|████████████████████████████████████████████████████| 391/391 [00:17<00:00, 22.03it/s, v_num=0, loss=0.0428, accuracy=1.000, f1_score=1.000, val_loss=0.128]`Trainer.fit` stopped: `max_epochs=3` reached.                                                                                                                            
# End train!
# Epoch 2: 100%|████████████████████████████████████████████████████| 391/391 [00:17<00:00, 22.02it/s, v_num=0, loss=0.0428, accuracy=1.000, f1_score=1.000, val_loss=0.128]
# Validation DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 79/79 [00:01<00:00, 48.27it/s]
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#      Validate metric           DataLoader 0
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#         val_loss            0.09857720881700516
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 79/79 [00:01<00:00, 47.90it/s]
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#         test_loss           0.11288387328386307
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

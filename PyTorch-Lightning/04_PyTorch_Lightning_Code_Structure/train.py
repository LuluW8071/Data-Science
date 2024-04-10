import torch
import pytorch_lightning as pl

from model import neuralnet
from dataset import MNISTDataModule
import config

if __name__ == "__main__":
    # Initialize the MNISTDataModule Dataloader 
    dataloader = MNISTDataModule(data_dir=config.DATA_DIR, 
                                batch_size=config.BATCH_SIZE, 
                                num_workers=config.NUM_WORKERS)

    # Initialize the model
    model = neuralnet(config.INPUT_SIZE, 
                      config.HIDDEN_SIZE, 
                      config.NUM_CLASSES).to(config.ACCELERATOR)

    # Create a Trainer instance for managing the training process.
    trainer = pl.Trainer(accelerator=config.ACCELERATOR,
                        devices=config.DEVICES,
                        min_epochs=1,
                        max_epochs=config.NUM_EPOCHS,
                        precision=config.PRECISION)

    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)
    trainer.test(model, dataloader)



# DEMO RESULTS:
# ======================================================================================

# /usr/local/lib/python3.10/dist-packages/lightning_fabric/connector.py:563: `precision=16` is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
# Using 16bit Automatic Mixed Precision (AMP)
# GPU available: True (cuda), used: True
# TPU available: False, using: 0 TPU cores
# IPU available: False, using: 0 IPUs
# HPU available: False, using: 0 HPUs
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

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
# Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
#   self.pid = os.fork()
# Epoch 0: 100% 391/391 [00:08<00:00, 48.27it/s, v_num=0, loss=0.118, accuracy=0.988, f1_score=0.988]
# Validation: |          | 0/? [00:00<?, ?it/s]
# Validation:   0% 0/79 [00:00<?, ?it/s]       
# Validation DataLoader 0:   0% 0/79 [00:00<?, ?it/s]
# Validation DataLoader 0:  25% 20/79 [00:00<00:00, 69.95it/s]
# Validation DataLoader 0:  51% 40/79 [00:00<00:00, 65.04it/s]
# Validation DataLoader 0:  76% 60/79 [00:00<00:00, 63.82it/s]
# Validation DataLoader 0: 100% 79/79 [00:01<00:00, 65.00it/s]
# Epoch 1: 100% 391/391 [00:08<00:00, 44.20it/s, v_num=0, loss=0.104, accuracy=0.950, f1_score=0.950]
# Validation: |          | 0/? [00:00<?, ?it/s]
# Validation:   0% 0/79 [00:00<?, ?it/s]       
# Validation DataLoader 0:   0% 0/79 [00:00<?, ?it/s]
# Validation DataLoader 0:  25% 20/79 [00:00<00:00, 75.62it/s]
# Validation DataLoader 0:  51% 40/79 [00:00<00:00, 67.74it/s]
# Validation DataLoader 0:  76% 60/79 [00:00<00:00, 66.67it/s]
# Validation DataLoader 0: 100% 79/79 [00:01<00:00, 67.01it/s]
# Epoch 2: 100% 391/391 [00:08<00:00, 44.09it/s, v_num=0, loss=0.123, accuracy=0.938, f1_score=0.938] 
# Validation: |          | 0/? [00:00<?, ?it/s]
# Validation:   0% 0/79 [00:00<?, ?it/s]       
# Validation DataLoader 0:   0% 0/79 [00:00<?, ?it/s]
# Validation DataLoader 0:  25% 20/79 [00:00<00:00, 73.54it/s]
# Validation DataLoader 0:  51% 40/79 [00:00<00:00, 67.29it/s]
# Validation DataLoader 0:  76% 60/79 [00:00<00:00, 66.01it/s]
# Validation DataLoader 0: 100% 79/79 [00:01<00:00, 67.00it/s]
# Epoch 3: 100% 391/391 [00:08<00:00, 44.02it/s, v_num=0, loss=0.156, accuracy=0.938, f1_score=0.938] 
# Validation: |          | 0/? [00:00<?, ?it/s]
# Validation:   0% 0/79 [00:00<?, ?it/s]       
# Validation DataLoader 0:   0% 0/79 [00:00<?, ?it/s]
# Validation DataLoader 0:  25% 20/79 [00:00<00:00, 70.92it/s]
# Validation DataLoader 0:  51% 40/79 [00:00<00:00, 67.21it/s]
# Validation DataLoader 0:  76% 60/79 [00:00<00:00, 65.85it/s]
# Validation DataLoader 0: 100% 79/79 [00:01<00:00, 67.37it/s]
# Epoch 4: 100% 391/391 [00:07<00:00, 49.50it/s, v_num=0, loss=0.138, accuracy=0.962, f1_score=0.962] 
# Validation: |          | 0/? [00:00<?, ?it/s]
# Validation:   0% 0/79 [00:00<?, ?it/s]       
# Validation DataLoader 0:   0% 0/79 [00:00<?, ?it/s]
# Validation DataLoader 0:  25% 20/79 [00:00<00:01, 46.04it/s]
# Validation DataLoader 0:  51% 40/79 [00:00<00:00, 45.16it/s]
# Validation DataLoader 0:  76% 60/79 [00:01<00:00, 43.37it/s]
# Validation DataLoader 0: 100% 79/79 [00:01<00:00, 44.49it/s]
# Epoch 4: 100% 391/391 [00:09<00:00, 39.56it/s, v_num=0, loss=0.138, accuracy=0.962, f1_score=0.962]`Trainer.fit` stopped: `max_epochs=5` reached.
# Epoch 4: 100% 391/391 [00:09<00:00, 39.49it/s, v_num=0, loss=0.138, accuracy=0.962, f1_score=0.962]
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# Validation DataLoader 0: 100% 79/79 [00:01<00:00, 66.51it/s]
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃      Validate metric      ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         val_loss          │    0.06699983030557632    │
# └───────────────────────────┴───────────────────────────┘
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# Testing DataLoader 0: 100% 79/79 [00:01<00:00, 71.34it/s]
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         test_loss         │    0.08792530000209808    │
# └───────────────────────────┴───────────────────────────┘

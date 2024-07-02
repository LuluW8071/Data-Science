""" Callback """
# Callback in computer programming is a funciton that is stored as data and designed to be called 
# by another function. In lightning, it allows you to add arbitrary self-contained programs to your training.

# NOTE: Reference link: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html

# =======================================================
""" EarlyStopping Callback """ 
# This callback can be used to monitor a metrci and stop the training when no improvement is observed.
# Further more stopping threshold or divergence threshold can be added for 
# stopping training immediately once the monitored quantity reaches certain threshold or
# monitored quantity becomes worse than certain threshold
# Reference Link: https://lightning.ai/docs/pytorch/stable/common/early_stopping.html#earlystopping-callback

""" ModelCheckpoint Callback """
# This callback saved the model checkpoints at every epoch by default.
# Through this callback, last saved checkpoints can also be retireved and trained further upon.
# Reference Link: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint

# =======================================================

# NOTE: Other Built-in Callbacks: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#built-in-callbacks
# =======================================================

""" Custom Callback Class """
# The callback hooks are implemented in train.py

from pytorch_lightning.callbacks import Callback

class printCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Start train!")

    def on_train_end(self, trainer, pl_module):
        print("End train!")

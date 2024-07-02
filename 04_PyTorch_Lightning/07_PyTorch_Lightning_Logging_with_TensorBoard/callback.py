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

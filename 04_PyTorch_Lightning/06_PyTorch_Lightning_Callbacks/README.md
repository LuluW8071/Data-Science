## PyTorch Lightning Callbacks

In PyTorch Lightning, **callbacks** offer a way to insert custom logic at key points in the training, validation, and testing process, enabling automation and customization without cluttering the core model code. They facilitate tasks like model checkpointing, early stopping, and dynamic learning rate adjustments, making routine processes automatic and error-free. By monitoring performance metrics, callbacks can also trigger actions like saving the best model or halting training when improvements cease.

Some of the commonly used PyTorch Lightning Callbacks hooks are: 

| Callback                   | Description                                                                                         |
|----------------------------|-----------------------------------------------------------------------------------------------------|
| **ModelCheckpoint**        | Automatically saves your model based on a monitored metric like validation loss, configurable to save the best model, last model, or periodically. |
| **EarlyStopping**          | Stops training when a monitored metric stops improving, preventing overfitting and saving resources. |
| **LearningRateMonitor**    | Logs the learning rate during training, useful for observing changes, especially with adaptive schedulers. |
| **ProgressBar**            | Built-in callback managing the progress bar, customizable to show additional information. |
| **Stochastic Weight Averaging (SWA)** | Averages model weights over the last epochs for a smoother, potentially more generalizable model. |

For more details on built-in callbacks, [visit here](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html)
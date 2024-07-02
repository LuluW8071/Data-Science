## PyTorch Lightning Logging with TensorBoard

Install TensorBoard:

```bash
pip install tensorboard
```

In PyTorch Lightning, **logging with TensorBoard** offers a way to view the metrics in a graphical grid view. You can also log images at specified steps, batches, or epochs. This is helpful for visualizing the training process instead of manually reading numeric metrics. Furthermore, you can view the model graphs, histograms and many more things.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*1e6feu4blFUBJTQCQ9XM9Q.gif">

To view TensorBoard on your localhost, run one of the following commands:

```bash
tensorboard --logdir=tb_logs --bind_all
```

or

```bash
tensorboard --logdir tb_logs
```

### Other Common Metrics Loggers:

1. MLFlow
2. Neptune
3. CometML

For more details on TensorBoard, [visit here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html).
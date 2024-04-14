## PyTorch Lightning Code Structure

This chapter outlines how to structure a PyTorch Lightning project into modular components for better organization and maintainability.

### Pros of Modular Structure

| Pros                              |
|-----------------------------------|
| **Modularity**: Easier management and understanding of components. |
| **Reusability**: Promotes code reuse and reduces redundancy. |
| **Scalability**: Simplifies project growth and maintenance. |
| **Collaboration**: Enhances teamwork with clear module responsibilities. |

### Modular Structure

| File            | Description                                                   |
|-----------------|---------------------------------------------------------------|
| `config.py`     | Contains hyperparameters and configurations for the model.    |
| `dataset.py`    | Handles data loading and preprocessing, creating data loaders.|
| `model.py`      | Defines the PyTorch Lightning model and training/validation steps. |
| `train.py`      | Orchestrates the training process, leveraging other modules.  |

### Train Command

```python
py train.py
```
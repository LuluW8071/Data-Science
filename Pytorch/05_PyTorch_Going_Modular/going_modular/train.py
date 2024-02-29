import os 
import torch 
import data_setup
import dataset, engine, model
import utils

from torchvision import transforms

# Setup Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
NUM_EPOCHS = 10
HIDDEN_UNITS = 64
LEARNING_RATE = 0.001

# Setup directories
train_dir = "dataset/train"
test_dir = "dataset/test"

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create transformation for image
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.TrivialAugmentWide(num_magnitude_bins=21),
    transforms.ToTensor()
])

# Create DataLoaders through data_setup.py
train_dataloader, test_dataloader, class_names = dataset.create_dataloaders(
    train_dir, test_dir, data_transform, BATCH_SIZE, NUM_WORKERS
)

# Create Model through model.py
model = model.TinyVGGModel(input_shape=3,
                           hidden_units=64,
                           output_shape=len(class_names)).to(device)

# img_sample = torch.rand(1, 3, 128, 128)
# print(model(img_sample.to(device)))

# Setup loss_fn and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start the training through engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model through utils.py 
utils.save_model(model=model,
                 target_dir="models",
                 model_name="tinyvgg_model.pth")





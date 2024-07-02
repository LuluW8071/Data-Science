import requests
import os 
from pathlib import Path
import torch
import torchvision 
import torchvision.transforms as transforms

# Setting device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def download_image(url_or_path, destination_path):
    """
    Download the image from the URL or copy from local path.
    """
    # Check if the provided URL is a local path
    if os.path.exists(url_or_path): 
        with open(url_or_path, "rb") as f_in:
            with open(destination_path, "wb") as f_out:
                f_out.write(f_in.read())
        print(f"Image copied successfully to: {destination_path}")
    else:
        # Web URL
        request = requests.get(url_or_path)
        with open(destination_path, "wb") as f:
            f.write(request.content)
        print(f"Image downloaded successfully to: {destination_path}")


def load_and_preprocess_image(image_path):
    """
    Load and preprocess the image from the specified path.
    """
    custom_image_uint8 = torchvision.io.read_image(str(image_path))
    custom_image = custom_image_uint8.type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    custom_image = custom_image / 255.
    custom_image_transform = transforms.Compose([
        transforms.Resize((128, 128), antialias=True) # set antialias = True to remove warning
    ])

    # Apply transformations
    custom_image_transformed = custom_image_transform(custom_image)
    return custom_image_transformed

def predict_image(model, image_tensor, class_names):
    """
    Predict the class label for the given image tensor using the provided model.
    """
    with torch.inference_mode():
        # Add an extra dimension to image
        image_tensor_with_batch_size = image_tensor.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension
        image_pred = model(image_tensor_with_batch_size.to(device))
    image_pred_probs = torch.softmax(image_pred, dim=1)
    # Convert prediction probabilities -> prediction labels
    image_pred_label = torch.argmax(image_pred_probs, dim=1)
    return class_names[image_pred_label.numpy()], image_pred_probs
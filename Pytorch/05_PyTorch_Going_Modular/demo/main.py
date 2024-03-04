
import torch
import numpy as np
import sys
from os.path import join, realpath
from PIL import Image
from pathlib import Path
import streamlit as st
from image_loader import *

sys.path.append(realpath(join(realpath(__file__), '..', '..')))
from going_modular.model import TinyVGGModel


def main():
    st.title("Food Classifier")

    # Setting device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # st.write(f"Device: {device}")

    # Load Trained Model
    MODEL_SAVE_PATH = "../models/tinyvgg_model.pth"
    model_info = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))

    # Instantiate Model
    model = TinyVGGModel(
        input_shape=3,
        hidden_units=64,
        output_shape=8).to(device)

    # Define paths
    data_path = Path("test_images/")

    # Image upload section
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image
        custom_image_path = data_path / uploaded_file.name
        with open(custom_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and preprocess the image
        custom_image_transformed = load_and_preprocess_image(custom_image_path)

        # Load the model
        model.load_state_dict(model_info)
        model.eval()

        # Predict the label for the image
        class_names = np.array(['Donuts', 'Dumplings', 'Ice Cream', 
                                'Pizza', 'Ramen', 'Samosa',
                                'Steak', 'Sushi'])
        predicted_label, image_pred_probs = predict_image(model,
                                                          custom_image_transformed,
                                                          class_names)


        # Prediction result section
        st.markdown(
            f'<h3 style="color: green;">Prediction Result</h3>', 
            unsafe_allow_html=True
        )

        # Create a two-column layout
        col1, col2 = st.columns([1, 3])

        # Display prediction label and confidence rate on the left column
        col1.write(f"Predicted Food: **{predicted_label[0]}**")
        col1.write(f"Confidence: **{image_pred_probs.max()* 100:.2f}%**")

        # Display the uploaded image on the right column
        with col2:
            image = Image.open(custom_image_path)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
if __name__ == "__main__":
    main()
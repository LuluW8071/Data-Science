""" Download and Extract Food_dataset.zip from google drive"""
# NOTE: The Dataset was downloaded from:
# https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

# Among 101 classes of foods only 8 were choosen:
# 1. Pizza
# 2. Steak 
# 3. Donuts
# 4. Dumplings
# 5. Ice_Cream
# 6. Ramen
# 7. Samosa
# 8. Sushi
# The 8 classes of food_dataset was then compressed using zip

# NOTE: Notebook for getting food_dataset from Food101 can be found on the link below:
# https://github.com/LuluW8071/Data-Science/blob/main/Pytorch/04_PyTorch_Custom_Datasets/Extras/Pytorch_Custom_Food_Datasets.ipynb

import gdown
import zipfile
import os 

file_url = "https://drive.google.com/uc?id=1J0syU84FNmtxkf9AzDPdRSDmtUr1CSy8"
file_name = "Food_dataset.zip"

# Download the file from google drive
gdown.download(file_url, file_name, quiet = False)
extract_dir = './dataset'

# Extract the zip file
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Remove the zip file after extraction 
os.remove(file_name)
print("Files extracted successfully to:", extract_dir)

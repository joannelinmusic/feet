from path import my_path
import os
import pandas as pd
from PIL import Image

folder_path = os.path.join(my_path, '06192023 SFI renamed')
patient_path = os.path.join(folder_path, 'P001 SAGIR', 'MRI ANKLE (LEFT) W_O CONT_5891215')
distinct_dimensions = set()

def all_image_dimensions():
    # Iterate through subdirectories
    for root, subdirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                # Get the image file path
                image_path = os.path.join(root, file)
                
                # Open the image using PIL
                image = Image.open(image_path)
                
                # Get image dimensions (width, height, and channels)
                width, height = image.size
                channels = len(image.getbands())
                
                # Add the dimensions to the set
                distinct_dimensions.add((width, height, channels))

    # Print the distinct image dimensions as a single string
    for width, height, channels in distinct_dimensions:
        dimensions_string = f"{width}, {height}, {channels}"
        print(dimensions_string)


def get_single_patient_dimension():
    unique_dimensions = set()
    for filename in os.listdir(patient_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(patient_path, filename)
            with Image.open(image_path) as img:
                dimensions = img.size
                unique_dimensions.add(dimensions)

    print("Distinct image dimensions:")
    for width, height in unique_dimensions:
        print(f"{width} x {height}")

all_image_dimensions()
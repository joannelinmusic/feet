import os
import pandas as pd
from PIL import Image
from path import my_path

# Specify the directory path
folder_path = os.path.join(my_path, '06192023 SFI renamed')


# Get a list of all subdirectories
subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

attribute_names = sorted(set([subdir[5:] for subdir in subdirectories]))

# Extract image types (excluding the patient ID) from subdirectories
image_types = sorted(set([subdir[5:] for subdir in subdirectories]))

# Create a dictionary to store image type and dimension counts
count_dict = {}

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
            
            # Find the correct image type from the subdirectory
            image_type = None
            for subdir in subdirectories:
                if subdir[5:] in attribute_names and os.path.basename(root) == subdir:
                    image_type = subdir[5:]
                    break
            
            if image_type:
                # Create a dimensions tuple
                dimensions = (width, height, channels)
                
                # Increment the count for the dimensions and image type
                count_dict.setdefault((image_type, dimensions), 0)
                count_dict[(image_type, dimensions)] += 1

# Create a DataFrame from the count_dict
columns = ['image_type', 'dimension', 'total_count']
df_counts = pd.DataFrame(columns=columns)

# Populate the DataFrame with counts
for (image_type, dimensions), count in count_dict.items():
    df_counts = df_counts.append({'image_type': image_type, 'dimension': dimensions, 'total_count': count}, ignore_index=True)

# Write the DataFrame to a CSV file
counts_csv_path = os.path.join(my_path, 'image_type_dimension_counts.csv')
df_counts.to_csv(counts_csv_path, index=False)

print("Counts CSV written:", counts_csv_path)
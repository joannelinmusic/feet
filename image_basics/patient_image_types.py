from path import my_path
import os
import pandas as pd

folder_path = os.path.join(my_path, '06192023 SFI renamed')

subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

"""The following code writes the csv patient_image_types.csv that records each patient's image types"""
patient_ids = set([subdir[:5] for subdir in subdirectories])
attribute_names = sorted(set([subdir[5:] for subdir in subdirectories]))

# Create a DataFrame with columns
columns = ['patientID'] + attribute_names
df = pd.DataFrame(columns=columns)

# Populate the DataFrame with "yes" or "no" values
for patient_id in patient_ids:
    row_data = {'patientID': patient_id}
    for attribute_name in attribute_names:
        if patient_id + attribute_name in subdirectories:
            row_data[attribute_name] = 'yes'
        else:
            row_data[attribute_name] = 'no'
    df = df.append(row_data, ignore_index=True)

# Write the DataFrame to a CSV file
csv_path = os.path.join(my_path, 'image_basic_data', 'output.csv')
df.to_csv(csv_path, index=False)

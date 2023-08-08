from path import my_path
import os
import pandas as pd


csv_path = os.path.join(my_path, 'image_basic_data', 'patient_image_types.csv')

# Read the existing CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Exclude the 'patientID' column from attribute_names
attribute_names = [col for col in df.columns if col != 'patientID']

# Create a DataFrame to store attribute counts
count_df = pd.DataFrame(columns=['Image_type', 'Scans(count_by_patient)'])

# Calculate and populate attribute counts
for attribute_name in attribute_names:
    total_yes = (df[attribute_name] == 'yes').sum()
    count_df = count_df.append({'Image_type': attribute_name, 'Scans(count_by_patient)': total_yes}, ignore_index=True)

# Specify the path for the new CSV file
# count_csv_path = os.path.join(my_path, 'image_basic_data', 'type_counts.csv')

# Write the count DataFrame to a new CSV file
# count_df.to_csv(count_csv_path, index=False)

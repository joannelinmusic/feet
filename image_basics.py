from path import my_path
import os
import pandas as pd

folder_path = os.path.join(my_path, '06192023 SFI renamed')


"""The following code prints out the distinct folder name of images"""
subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]
distinct_names = set([subdir[5:] for subdir in subdirectories])
for name in distinct_names:
    print(name)

"""The following code writes the csv patient_image_types.csv that records each patient's image types"""
# patient_ids = set([subdir[:5] for subdir in subdirectories])
# attribute_names = sorted(set([subdir[5:] for subdir in subdirectories]))

# # Create a DataFrame with columns
# columns = ['patientID'] + attribute_names
# df = pd.DataFrame(columns=columns)

# # Populate the DataFrame with "yes" or "no" values
# for patient_id in patient_ids:
#     row_data = {'patientID': patient_id}
#     for attribute_name in attribute_names:
#         if patient_id + attribute_name in subdirectories:
#             row_data[attribute_name] = 'yes'
#         else:
#             row_data[attribute_name] = 'no'
#     df = df.append(row_data, ignore_index=True)

# # Write the DataFrame to a CSV file
# csv_path = os.path.join(my_path, 'image_basic_data', 'output.csv')
# df.to_csv(csv_path, index=False)



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
count_csv_path = os.path.join(my_path, 'image_basic_data', 'try.csv')

# Write the count DataFrame to a new CSV file
count_df.to_csv(count_csv_path, index=False)

print("Attribute count CSV written:", count_csv_path)
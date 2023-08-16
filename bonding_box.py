import os
from path import my_path
import pandas as pd
from PIL import Image

def crop_the_box():
    df = pd.read_csv(csv_path)
    print(df.head())
    for root, dirs, files in os.walk(folder_path):
        if files:
            patient = os.path.basename(os.path.dirname(root))  
            dimensions_set = set()
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename in df['image'].tolist():
                    image_path = os.path.join(root, filename)
                    
                    patient_id = patient[0:4]

                    x = df.loc[df['image'] == filename, 'value_x'].values[0]
                    y = (df.loc[df['image'] == filename, 'original_height'].values[0]) - (df.loc[df['image'] == filename, 'value_y'].values[0])
                    width = df.loc[df['image'] == filename, 'value_width'].values[0]
                    height = df.loc[df['image'] == filename, 'value_height'].values[0]

                    image = Image.open(image_path)

                    cropped_image = image.crop((x, y - height, x + width, y))


                    #save the cropped image
                    cropped_file_path = os.path.join(my_path, 'bonding_box', patient_id, filename[0:14])
                    counter = 1
                    while os.path.exists(cropped_file_path + f"_{counter:02}.jpg"):
                        counter += 1
                    cropped_file_path = cropped_file_path + f"_{counter:02}.jpg"
                    cropped_image.save(cropped_file_path)   

    
def bonding_box_folder():
    df = pd.read_csv(csv_path)
    df_patients = df['image'].apply(lambda x: x[0:4]).unique().tolist()
    for patient in df_patients:
        folder_path = os.path.join(my_path, 'bonding_box', patient)
        os.makedirs(folder_path, exist_ok=True)

def main():
    bonding_box_folder()
    crop_the_box()
    

if __name__ == '__main__':
    csv_path = os.path.join(my_path, 'json_data', 'sag_IR_json_columns', 'annotations', 'results_box_cleaned.csv')
    folder_path = os.path.join(my_path, '06192023 SFI renamed')
    image_path = os.path.join(folder_path, 'P001 SAGIR', 'MRI ANKLE (LEFT) W_O CONT_5891215', 'P001 SAGIR_010.jpg')

    main()

    
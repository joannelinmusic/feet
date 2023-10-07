from path import my_path
import os
import pandas as pd
from PIL import Image
from shutil import copyfile
import random


def copy_image():
    df = pd.read_csv(csv_path)

    # Iterate through rows in the DataFrame
    for index, row in df.iterrows():
        image_name, tag = row['image'], row['tag']

        for patientNumber in os.listdir(image_dir):
            if 'SAGT1' in patientNumber:
                for last_folder in os.listdir(os.path.join(image_dir, patientNumber)):
                    if not last_folder.startswith('.') and os.path.isdir(os.path.join(image_dir, patientNumber, last_folder)):
                        for slices in os.listdir(os.path.join(image_dir, patientNumber, last_folder)):
                            if slices == image_name:
                                if random.random() < 0.7:
                                    dest_dir = os.path.join(train_dir)
                                elif random.random() < 0.8:
                                    dest_dir = os.path.join(val_dir)
                                else:
                                    dest_dir = os.path.join(test_dir)
                                
                                if tag == "Positive":
                                    dest_dir = os.path.join(dest_dir, "Positive")
                                elif tag == "Negative":
                                    dest_dir = os.path.join(dest_dir, "Negative")
                                else:
                                    continue
                                
                                src_path = os.path.join(image_dir, patientNumber, last_folder, image_name)
                                print(src_path)
                                dest_path = os.path.join(folder_path, dest_dir, image_name)
                                
                                # Copy the image from the source to the destination directory
                                copyfile(src_path, dest_path)


def create_fodler():
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for directory in ["Positive", "Negative"]:
        train_subdir = os.path.join(train_dir, directory)
        val_subdir = os.path.join(val_dir, directory)
        test_subdir = os.path.join(test_dir, directory)
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(val_subdir, exist_ok=True)
        os.makedirs(test_subdir, exist_ok=True)


def main():
    # create_fodler()
    copy_image()

if __name__ == '__main__':
    image_dir = os.path.join(my_path, '06192023 SFI renamed')
    folder_path = os.path.join(my_path, '864')
    csv_path = os.path.join(my_path, 'image_basic_data', '864_slices.csv')
    train_dir = os.path.join(folder_path, "train")
    val_dir = os.path.join(folder_path, "validate")
    test_dir = os.path.join(folder_path, "test")
    main()

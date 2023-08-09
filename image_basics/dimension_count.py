import os
import pandas as pd
from PIL import Image
from path import my_path
from distinct_image_types import distinct_image_types


def patient_image_dimension():
    folder_dimension_list = []
    for root, dirs, files in os.walk(folder_path):
        if files:
            patient = os.path.basename(os.path.dirname(root))  
            dimensions_set = set()
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_file = os.path.join(root, filename)
                    with Image.open(image_file) as img:
                        dimensions = img.size
                        dimensions_set.add(dimensions)
            
            folder_dimension_list.extend([(patient[0:4], patient[5:], dimensions) for dimensions in dimensions_set])

    df = pd.DataFrame(folder_dimension_list, columns=['patient', 'type', 'dimensions'])
    df.to_csv(output_csv_path, index=False)

def type_dimension_count():
    data = pd.read_csv(output_csv_path)
    grouped = data.groupby(['type', 'dimensions']).size().reset_index(name='count')
    grouped.to_csv(type_outout_path, index=False)

def main():
    # patient_image_dimension()
    type_dimension_count()

if __name__ == "__main__":
    folder_path = os.path.join(my_path, '06192023 SFI renamed')
    output_csv_path = os.path.join(my_path, 'image_basic_data', 'patient_dimensions.csv')
    type_outout_path = os.path.join(my_path, 'image_basic_data', 'dimensions_count.csv')
    image_types = distinct_image_types()
    main()
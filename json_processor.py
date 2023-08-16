from path import my_path
import pandas as pd
import os
import ast

def attributes_json():
    
    df = pd.read_json(file_path)

    
    columns = df.columns
    columns.to_series().to_csv(output_csv_path_columns, index=False)

    csv_dir = os.path.join(my_path, 'json_data', 'sag_IR_json_columns')
    os.makedirs(csv_dir, exist_ok=True)

    # Iterate through each column
    for column in columns:
        # Create a DataFrame containing the 'id', column name, and column values
        column_df = pd.DataFrame({
            'ids': df['id'],
            'column_name': column,
            'column_values': df[column]
        })
        
        # Create a CSV file path for the current column
        column_csv_path = os.path.join(csv_dir, f'{column}.csv')
        
        # Write the column data to the CSV file
        column_df.to_csv(column_csv_path, index=False)

def annotations():
    df = pd.read_json(file_path)
    csv_dir = os.path.join(my_path, 'json_data', 'annotations_columns')
    annotations_column = df['annotations']
    print(annotations_column.head())
    
def id_result_box_raw():
    df = pd.read_json(file_path)
    data_df = pd.DataFrame(columns=['file_upload', 'results'])
    for index, row in df.iterrows():
        data_df = data_df.append({
            'file_upload': row['file_upload'],
            'results': row['annotations'][0]['result'], 
            
        }, ignore_index=True)
    data_df.to_csv(output_csv_path_annotations_results, index=False)

def id_result_clean():
    df = pd.read_csv(results_raw)
    clean_df = pd.DataFrame(columns=['image', 'original_width', 'original_height', 'image_rotation', 'value_x', 'value_y', 'value_width', 'value_height', 'value_rotation', 'value_rectanglelabels'])
    for index, row in df.iterrows():
        if len(ast.literal_eval(row['results']))>0:
            for box in range(len(ast.literal_eval(row['results']))):
                clean_df = clean_df.append({
                    'image': row['image'],
                    'original_width': ast.literal_eval(row['results'])[box]['original_width'], 
                    'original_height': ast.literal_eval(row['results'])[box]['original_height'], 
                    'image_rotation': ast.literal_eval(row['results'])[box]['image_rotation'], 
                    
                    'value_x': ast.literal_eval(row['results'])[box]['value']['x'], 
                    'value_y': ast.literal_eval(row['results'])[box]['value']['y'], 
                    'value_width': ast.literal_eval(row['results'])[box]['value']['width'], 
                    'value_height': ast.literal_eval(row['results'])[box]['value']['height'], 
                    'value_rotation': ast.literal_eval(row['results'])[box]['value']['rotation'], 
                    'value_rectanglelabels': ast.literal_eval(row['results'])[box]['value']['rectanglelabels'], 
                    
                }, ignore_index=True)
    clean_df.to_csv(output_csv_path_annotations_results, index=False)

def box_rate():
    box = 0
    null_box =1
    df = pd.read_csv(results_raw)
    for index, row in df.iterrows():
        if len(ast.literal_eval(row['results']))>0:
            box += 1
        else:
            null_box += 1
    
    print('Number of images with bonding boxs:', box)
    print('Number of images without bonding box:', null_box)

def main():
    # attributes_json()
    # annotations()
    # id_result_box_raw()
    # id_result_clean()
    box_rate()

if __name__ == '__main__':
    file_path = os.path.join(my_path, 'SagIRAnkle2.0.json')

    output_csv_path_columns = os.path.join(my_path, 'json_data', 'sag_IR_json_columns.csv')
    output_csv_path_annotations_results = os.path.join(my_path, 'json_data', 'sag_IR_json_columns', 'annotations', 'results.csv')
    results_raw = os.path.join(my_path, 'json_data', 'sag_IR_json_columns', 'annotations', 'results_raw(image_name_aligned).csv')
    annotations_data_path = os.path.join(my_path, 'json_data', 'sag_IR_json_columns', 'annotations.csv')
    main()
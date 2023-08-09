from path import my_path
import pandas as pd
import os

def attributes_json():
    
    df = pd.read_json(file_path)

    
    columns = df.columns
    columns.to_series().to_csv(output_csv_path, index=False)

    csv_dir = os.path.join(my_path, 'json_data', 'sag_T1_json_columns')
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

def main():
    attributes_json()

if __name__ == '__main__':
    file_path = os.path.join(my_path, 'Sag T1 Ankle Ricky 1-115.json')
    output_csv_path = os.path.join(my_path, 'json_data', 'sag_T1_json_columns.csv')
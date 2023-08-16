from path import my_path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic
import pandas as pd
import os

def pie_chart_rectanglelabels():
    df = pd.read_csv(csv_path)
    binary_variable_counts = df['value_rectanglelabels'].value_counts()
    
    plt.pie(binary_variable_counts, labels=binary_variable_counts.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Label Distribution')
    plt.show()

def main():
    pie_chart_rectanglelabels()

if __name__ == '__main__':
    csv_path = os.path.join(my_path, 'json_data', 'sag_IR_json_columns', 'annotations', 'results_box_cleaned.csv')
    
    main()

    

from path import my_path
import os
import pandas as pd

folder_path = os.path.join(my_path, '06192023 SFI renamed')


"""The following code prints out the distinct folder name of images"""
subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]
distinct_names = set([subdir[5:] for subdir in subdirectories])
for name in distinct_names:
    print(name)





import csv
import pandas as pd
from pathlib import Path

directory_in_str = "/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results"

pathlist = Path(directory_in_str).glob('**/data_output.csv')
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)

    with open(path_in_str, mode='w') as csv_file:
        with open(path_in_str, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            next(csv_reader)
            
            for row in csv_reader:
                
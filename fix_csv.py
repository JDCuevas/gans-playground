import csv
import pandas as pd
from pathlib import Path

directory_in_str = "/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results"

pathlist = Path(directory_in_str).glob('**/data_output.csv')
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)

    with open(path_in_str + '_fixed', mode='w') as fixed_csv_file:
        fixed_writer = csv.writer(fixed_csv_file, delimiter=',')

        with open(path_in_str, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            next(csv_reader)
            
            print(path_in_str)
            for row in csv_reader:
                fixed_r = row['generated'].strip('[]').strip().split(' ')
                fixed_r = [float(n) for n in fixed_r if n != '']
                
                fixed_writer.writerow(fixed_r)
                
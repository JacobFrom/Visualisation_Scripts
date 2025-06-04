import numpy as np
import os


datasets = os.listdir('G:\ARESTOR_DATA\dorchester_active_jf')

for dataset in datasets:
    experiment_dir = dataset
    print(experiment_dir)

print('finished iterating through datasets')
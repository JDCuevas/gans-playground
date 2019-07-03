import pandas as pd

from data_generation import mnist, stress_strain
from utils import plotting

pred = pd.read_csv('/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gans_playground/results/models/dnn_1d/stress_strain/BS_256/CI_5/ND_3/L_0.1/data_output.csv').values

samples, strains = stress_strain.StressStrainDS().generate_samples(0.02, 10, 10000)

plotting.plot_all_hist(x_values=strains, y_values=samples, y_values_2=pred, img_name='data_dist_histograms')

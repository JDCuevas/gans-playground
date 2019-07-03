import os
import pandas as pd
from utils import plotting
import matplotlib.pyplot as plt
import numpy as np
from data_generation import stress_strain as ss
from ast import literal_eval

def plot_losses(target, values, path="./losses_per_epoch.csv"):
    model_dir = path
    substring_list = ['BS', 'MD', 'CI', 'ND', 'L']
    substring_list.remove(target)

    settings_list = model_dir.split("/")
    settings = "".join(['-' + s for s in settings_list if any(substring in s for substring in substring_list)])

    path_index = model_dir.index(target + '_')
    path_beg = model_dir[:path_index]
    path_end = model_dir[path_index:][model_dir[path_index:].index("/"):] if len(model_dir[path_index:].split("/")) > 1 else ""

    plt.figure(figsize=(10, 10))

    for value in values:
        target_folder = target + '_' +  value
        p = path_beg + target_folder + path_end

        df = pd.read_csv(p)
        disc_losses = df['discriminator loss']
        gen_losses = df['generator loss']
        
        plotting.plot_across([disc_losses, gen_losses], ["Discriminator Loss " + target_folder, "Generator Loss " + target_folder], "epoch", "loss", "Losses per Epoch ")
    
    # Save
    if not os.path.exists(os.path.dirname('./visualization/' + target + '/')):
            try:
                os.makedirs(os.path.dirname('./visualization/' + target + '/'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    plt.savefig('./visualization/' + target + '/loss_per_epoch' + settings + '.png')
    plt.close()

def plot_dist(target, values, x_values, samples, path):
    model_dir = path
    model = model_dir[model_dir.index('/models/') + len('/models/'):model_dir.index('/mnist')]

    substring_list = ['BS', 'MD', 'CI', 'ND', 'L']
    substring_list.remove(target)

    settings_list = model_dir.split("/")
    settings = "".join(['-' + s for s in settings_list if any(substring in s for substring in substring_list)])

    path_index = model_dir.index(target + '_')
    path_beg = model_dir[:path_index]
    path_end = model_dir[path_index:][model_dir[path_index:].index("/"):] if len(model_dir[path_index:].split("/")) > 1 else ""

    f, axes = plt.subplots(int(len(x_values)/2), int(len(x_values) /
                                                     (len(x_values)/2)), figsize=(15, 15), sharex=False)

    for value in values:
        target_folder = target + '_' +  value
        p = path_beg + target_folder + path_end

        pred = pd.read_csv(p).values
        plotting.plot_hist_across(x_values=x_values, y_values=pred, y_label='stresses ' + target_folder, axes=axes)

    plotting.plot_hist_across(x_values=x_values, y_values=samples, y_label='stresses real', axes=axes, color='blue')

    plt.legend()
    plt.tight_layout()

    # Save
    if not os.path.exists(os.path.dirname('./visualization/' + target + '/')):
            try:
                os.makedirs(os.path.dirname('./visualization/' + target + '/'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    plt.savefig('./visualization/' + target + '/data_dist' + settings + '.png')
    plt.close()

if __name__ == '__main__':
    #plot_losses(target='BS', values=['16', '32', '64', '128'], path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/metrics_per_epoch.csv")
    #plot_losses(target='MD', values=['64', '128', '256', '512'], path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/metrics_per_epoch.csv")
    #plot_losses(target='CI', values=['4', '5', '6', '7'], path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/metrics_per_epoch.csv")
    #plot_losses(target='ND', values=['2', '3', '4', '5'], path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/metrics_per_epoch.csv")
    #plot_losses(target='L', values=['0.01', '0.1', '1', '10'], path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/metrics_per_epoch.csv")

    #num_examples = 10000
    #samples, x_values = ss.StressStrainDS().generate_samples(0.02, 10, num_examples)

    #plot_dist(target='BS', values=['16', '32', '64', '128'], x_values=x_values, samples=samples, path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/data_output.csv_fixed")
    #plot_dist(target='MD', values=['64', '128', '256', '512'], x_values=x_values, samples=samples, path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/data_output.csv_fixed")
    #plot_dist(target='CI', values=['4', '5', '6', '7'], x_values=x_values, samples=samples, path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/data_output.csv_fixed")
    #plot_dist(target='ND', values=['2', '3', '4', '5'], x_values=x_values, samples=samples, path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/data_output.csv_fixed")
    #plot_dist(target='L', values=['0.01', '0.1', '1', '10'], x_values=x_values, samples=samples, path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gan_test_problem/results/models/BS_16/MD_512/CI_5/ND_3/L_0.01/data_output.csv_fixed")

    plot_losses(target='L', values=['10.0'], path="/home/jcuevas7/Workspace/piganns-and-piml-summer-2019/gans_playground/results/models/dcgan_2d/mnist/BS_256/CI_1/ND_100/L_10.0/losses_per_epoch.csv")
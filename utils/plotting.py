import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot(variable, labels, x_label, y_label, title, img_name):
    for values, label in zip(variable, labels):
        plt.plot(values, label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_name + '.png')
    plt.close()


def plot_scatter(y_values, x_values, labels, x_label, y_label, title, img_name):
    for y_value, label in zip(y_values, labels):
        plt.scatter(np.expand_dims(x_values, -1),
                    np.expand_dims(y_value, -1), label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_name + '.png')
    plt.close()


def plot_hist(x_values, y_values, x_label, y_label, x, img_name):
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(16, 7))

    sns.violinplot(data=pd.DataFrame(
        data=y_values, columns=np.round(x_values, 3)), ax=ax_left)
    ax_left.set(xlabel=x_label, ylabel=y_label)

    itemindex = np.argmin(abs(x_values-x))
    sns.distplot(y_values[:, itemindex], bins=20)
    ax_right.set(xlim=(np.min(y_values), np.max(y_values)),
                 xlabel='stresses at strain of %.3f' % x_values[itemindex])
    plt.savefig(img_name + '.png')
    plt.close()


def plot_hist_comparison(x_values, y_values, y_values_2, x_label, y_label, x, img_name):
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(16, 7))

    sns.violinplot(data=pd.DataFrame(
        data=y_values, columns=np.round(x_values, 3)), ax=ax_left)
    ax_left.set(xlabel=x_label, ylabel=y_label)

    itemindex = np.argmin(abs(x_values-x))
    sns.distplot(y_values[:, itemindex], color="blue",
                 bins=20, label='y_values')
    if np.any(y_values_2):
        sns.distplot(y_values_2[:, itemindex],
                     color="red", bins=20, label='y_values_2')
    plt.legend()
    ax_right.set(xlim=(np.min(y_values), np.max(y_values)),
                 xlabel='stresses at strain of %.3f' % x_values[itemindex])
    plt.savefig(img_name + '.png')
    plt.close()


def plot_all_hist(x_values, y_values, y_values_2, img_name):
    f, axes = plt.subplots(int(len(x_values)/2), int(len(x_values) /
                                                     (len(x_values)/2)), figsize=(15, 15), sharex=False)
    for i in range(len(x_values)):
        sns.distplot(y_values[:, i], color="blue", bins=20, label='y_values', ax=axes[i % int(
            len(x_values)/2), 0 if i < len(x_values)/2 else 1])
        sns.distplot(y_values_2[:, i], color="red", bins=20, label='y_values_2', ax=axes[i % int(
            len(x_values)/2), 0 if i < len(x_values)/2 else 1])
        axes[i % int(len(x_values)/2), 0 if i < len(x_values)/2 else 1].set(xlim=(np.min(
            y_values), np.max(y_values)), xlabel='stresses at strain of %.3f' % x_values[i])

    plt.tight_layout()
    plt.savefig(img_name + '.png')
    plt.close()


def plot_individual_sample(sample, sample_output, n, img_name):
    fig = plt.figure(n)
    stress_ax = plt.plot(sample, sample_output)

    plt.xlabel('strain')
    plt.ylabel('stress')
    plt.title('Stress-Strain Curve ' + str(n + 1))
    plt.tight_layout()
    plt.savefig(img_name + '.png')
    plt.close()


def plot_violinplot(x_values, y_values, x_label, y_label, img_name):
    ax = sns.violinplot(data=pd.DataFrame(
        data=y_values, columns=np.round(x_values, 3)))
    ax.set(xlabel=x_label, ylabel=y_label)
    plt.savefig(img_name + '.png')
    plt.close()

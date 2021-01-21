import yaml
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

plt.style.use("ggplot")


def get_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_training(train_val_loss, train_val_acc, save=True):
    fig = plt.figure(figsize=(20, 8))

    ax_loss = fig.add_subplot(1, 2, 1)
    ax_loss.set_title("Loss")
    for phase, loss in train_val_loss.items():
        ax_loss.plot(loss, label=phase)
    ax_loss.legend()

    ax_acc = fig.add_subplot(1, 2, 2)
    ax_acc.set_title("Accuracy")
    for phase, acc in train_val_acc.items():
        ax_acc.plot(acc, label=phase)
    ax_acc.legend()

    if save:
        save_fig("loss_acc_plot", "images", fig_extension="png")


def save_fig(fig_name, fig_dir, tight_layout=True, fig_extension="png", resolution=300):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    path = os.path.join(fig_dir, fig_name + "." + fig_extension)
    print("Saving figure:", fig_name)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def confusion_matrix(model, y_pred, y_true, fig_size=10):

    stacked = torch.stack((y_true, y_pred), dim=1)
    confusion_matrix = torch.zeros(25, 25, dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    #delete the row coresponding to the j'th letter since it is out of the dataset because it involves motion
    confusion_matrix = np.delete(confusion_matrix, 9, axis=0)
    confusion_matrix = np.delete(confusion_matrix, 9, axis=1)

    # create the plot
    plt.figure(figsize = (fig_size,fig_size))
    ax = sns.heatmap(confusion_matrix, cmap= "Blues", linecolor = 'black' , linewidth = 0, annot = True, fmt='', xticklabels = letters, yticklabels = letters)
    ax.set(xlabel='Classified as', ylabel='True label')
    plt.show()
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
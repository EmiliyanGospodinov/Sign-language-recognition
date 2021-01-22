import yaml
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

plt.style.use("ggplot")


def get_config(config_file="config.yaml"):
    """
    Read config file which contains dataset paths, hyperparameter settings.

    Parameters
    ----------
    config_file : str, optional
        Path of config file, by default "config.yaml"

    Returns
    -------
    dict
        Configuration as dictionary
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed):
    """
    Fix random seed to ensure reproducibility

    Parameters
    ----------
    seed : int
        Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_training(train_val_loss, train_val_acc, save=True):
    """
    Plot loss and accuracy of model training

    Parameters
    ----------
    train_val_loss : dict
        Dictionary containing losses on training set and validation set
    train_val_acc : dict
        Dictionary containing accuracies on training set and validation set
    save : bool, optional
        Save plotting, by default True
    """
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


def save_fig(fig, fig_name, fig_dir, tight_layout=True, fig_extension="png", resolution=300):
    """
    Save figure

    Parameters
    ----------
    fig : Matplotlib.figure.Figure
        Figure to be saved
    fig_name : str
        Name of figure
    fig_dir : str
        Directory to save figure
    tight_layout : bool, optional
        Automatically adjusts subplot params so that the subplot(s) fits in to the figure area, by default True
    fig_extension : str, optional
        Extension of figure, by default "png"
    resolution : int, optional
        The resolution in dots per inch, by default 300
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    path = os.path.join(fig_dir, fig_name + "." + fig_extension)

    if tight_layout:
        plt.tight_layout()

    fig.savefig(path, format=fig_extension, dpi=resolution)
    print(f"Save {fig_name}.{fig_extension} in {fig_dir}")
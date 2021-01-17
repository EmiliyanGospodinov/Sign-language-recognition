import yaml
import os
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def get_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def plot_training(train_val_loss, train_val_acc, save_fig=True):
    train_loss = train_val_loss["train"]
    val_loss = train_val_loss["val"]
    train_acc = train_val_acc["train"]
    val_acc = train_val_acc["val"]

    fig = plt.figure(figsize=(20, 8))

    ax_loss = fig.add_subplot(1, 2, 1)
    ax_loss.set_title("Loss")
    ax_loss.plot(train_loss, label="Train")
    ax_loss.plot(val_loss, label="Validation")
    ax_loss.legend()

    ax_acc = fig.add_subplot(1, 2, 2)
    ax_acc.set_title("Accuracy")
    ax_acc.plot(train_acc, label="Train")
    ax_acc.plot(val_acc, label="Validation")
    ax_acc.legend()

    if save_fig:
        save_fig("loss_acc_plot", "images", fig_extension="png")


def save_fig(fig_name, fig_dir, tight_layout=True, fig_extension="png", resolution=300):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    path = os.path.join(fig_dir, fig_name + "." + fig_extension)
    print("Saving figure:", fig_name)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
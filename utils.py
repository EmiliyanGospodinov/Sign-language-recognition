import yaml
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

# imports needed for plotting the activation maps
import sign_language_mnist
from PIL import Image
import torchvision.transforms
import torchvision
import torchfunc


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
        save_fig(fig, "loss_acc_plot", "images", fig_extension="png")


def save_fig(
    fig, fig_name, fig_dir, tight_layout=True, padding=False, fig_extension="png", resolution=300, transparent=True
):
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
    padding: bool, optional
        Paddings around figure, by default False
    fig_extension : str, optional
        Extension of figure, by default "png"
    resolution : int, optional
        The resolution in dots per inch, by default 300
    transparent : bool, optional
        Use transparent background, by default True
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    path = os.path.join(fig_dir, fig_name + "." + fig_extension)

    if tight_layout:
        plt.tight_layout()

    if not padding:
        fig.savefig(
            path, format=fig_extension, dpi=resolution, bbox_inches="tight", pad_inches=0, transparent=transparent
        )
    else:
        fig.savefig(path, format=fig_extension, dpi=resolution, transparent=transparent)
    print(f"Save {fig_name}.{fig_extension} in {fig_dir}")


def confusion_matrix(y_pred, y_true, fig_size=10):
    """
    Creates a confusion matrix of a model to further analyse and visualize
    where the model has problems recognizing the right characters
    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor with the predictions made by the model
    y_true : torch.Tensor
        Tensor with the groud truth labels
    fig_size: int, optional
        Size of the matplotlib figure
    """
    stacked = torch.stack((y_true, y_pred), dim=1)
    confusion_matrix = torch.zeros(25, 25, dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    letters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
    ]

    # delete the row and column coresponding to the j'th letter since it is out of the dataset because it involves motion
    confusion_matrix = np.delete(confusion_matrix, 9, axis=0)
    confusion_matrix = np.delete(confusion_matrix, 9, axis=1)

    # create the plot
    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(
        confusion_matrix,
        cmap="Blues",
        linecolor="black",
        linewidth=0,
        annot=True,
        fmt="",
        xticklabels=letters,
        yticklabels=letters,
    )
    ax.set(xlabel="Classified as", ylabel="True label")
    plt.show()


def plot_activation_maps(model, img_dir="", layer_num=3):
    """
    Visualize the neural network activation maps with the usage of torchfunc library and hooks
    Parameters
    ----------
    model : CNN model
        CNN model whoes activation maps will be recorded and visualized
    img_dir : str
        Path to the input image
    layer_num: int, optional
        Number of layer whoes maps will be visualized,
        start at 3, previous visualize the image it self
    """
    cnn_model = torch.load(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # recorder saving inputs to all submodules
    recorder = torchfunc.hooks.recorders.ForwardPre()

    # register hooks for all submodules of the model
    # only a certain of them can be specified by index or layer type
    recorder.modules(cnn_model)

    # load the input image from the corresponding directory
    if img_dir == "":
        picture = sign_language_mnist.get_test_dataset()[0][0]
    else:
        try:
            transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.Scale((28, 28)),
                    transforms.ToTensor(),
                ]
            )

            image = transform(Image.open(img_dir))

        except:
            print("Wrong directory for the input image.")

    # push input image through the model
    output = cnn_model(picture.type(torch.FloatTensor).to(device).reshape(-1, 1, 28, 28))

    conv = recorder.data[layer_num][0].cpu().detach().numpy()
    size = recorder.data[layer_num][0].shape[1]

    fig = plt.figure(figsize=(20, 20))
    if size != 1:
        rows, columns = int(size / 8), 8
    else:
        rows, columns = 1, 1

    # create subfigure for every channel in the desired layer
    for i in range(rows * columns):
        fig.add_subplot(columns, rows, i + 1)
        plt.imshow(conv[0][i], cmap="gray")

    plt.show()


def plot_cnn_kernel(model, layer_num):
    """
    Visualize the neural network kernel weights
    IMPORTANT: this function will work only if it is used on layer which has weights/parameters !
    Parameters
    ----------
    model : CNN model
        CNN model whoes activation maps will be recorded and visualized
    layer_num: int
        Number of convolutional layer whoes weights will be visualized,
    """
    # extract the model features at the particular layer number
    layer = model.feature_extractor[layer_num]
    # getting the weight tensor data
    weight_tensor = model.feature_extractor[layer_num].weight.data

    nplots = weight_tensor.shape[0] * weight_tensor.shape[1]
    ncols = 12
    nrows = 1 + nplots // ncols

    npimg = np.array(weight_tensor.cpu().numpy(), np.float32)
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(weight_tensor.shape[0]):
        for j in range(weight_tensor.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(weight_tensor[i, j].cpu().numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + "," + str(j))
            ax1.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
    plt.tight_layout()
    plt.show()


def precision(y_pred, y_true, epsilon=1e-7):
    """
    calculate precision as evaluation metric
    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor with the predictions made by the model
    y_true : torch.Tensor
        Tensor with the groud truth labels
    epsilon: float
        float value for numerical stability
    Returns
    -------
    precision: float
        the calculate precision value
    """
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)

    return tp / (tp + fp + epsilon)


def recall(y_pred, y_true, epsilon=1e-7):
    """
    calculate recall as evaluation metric
    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor with the predictions made by the model
    y_true : torch.Tensor
        Tensor with the groud truth labels
    epsilon: float
        float value for numerical stability
    Returns
    -------
    recall: float
        the calculate recall value
    """
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    return tp / (tp + fn + epsilon)


def f1_score(y_pred, y_true, epsilon=1e-7):
    """
    calculate f1-score as evaluation metric
    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor with the predictions made by the model
    y_true : torch.Tensor
        Tensor with the groud truth labels
    epsilon: float
        float value for numerical stability
    Returns
    -------
    f1_score: float
        the calculate f1_score value
    """
    precision = precision(y_pred, y_true, epsilon)
    recall = recall(y_pred, y_true, epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1

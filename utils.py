import yaml
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

#imports needed for plotting the activation maps
from sign_language_mnist import get_test_dataset
from PIL import Image
import torchvision.transforms
import torchvision
import torchfunc


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


def plot_activation_maps(model, img_dir="", layer_num=3):

    cnn_model = torch.load(model) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # recorder saving inputs to all submodules
    recorder = torchfunc.hooks.recorders.ForwardPre()

    # register hooks for all submodules of the model
    # only a certain of them can be specified by index or layer type 
    recorder.modules(cnn_model)

    # load the input image from the corresponding directory
    if img_dir == "":
        picture = get_test_dataset()[0][0]
    else:
        try:
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Scale((28, 28)),
                transforms.ToTensor(),
            ])

            image = transform(Image.open(img_dir))

        except:
            print("Wrong directory for the input image.")

    # push input image through the model
    output = cnn_model(picture.type(torch.FloatTensor).to(device).reshape(-1, 1, 28, 28))

    conv = recorder.data[layer_num][0].cpu().detach().numpy()
    size = recorder.data[layer_num][0].shape[1]

    fig = plt.figure(figsize=(20, 20))
    if size != 1:
        rows, columns = int(size/8), 8
    else:
        rows, columns = 1,1
    
    #create subfigure for every channel in the desired layer
    for i in range(rows*columns):
        fig.add_subplot(columns, rows, i + 1)
        plt.imshow(conv[0][i], cmap='gray')

    plt.show()
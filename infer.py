import os
import torch
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np

map_characters = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])

def get_args_parser():
    parser = argparse.ArgumentParser(description="Model inference")
    parser.add_argument("model", type=str, help="Model path")
    parser.add_argument("img_dir", type=str, help="image path")
    return parser


def infer(model, img_dir):
    """
    Load an already trained model and infer an input image with it

    Parameters
    ----------
    model : str
        Path where the model is stored
    img_dir : str
        Path where the input image is stored

    Returns
    -------
    str:
        a character that corresponds to the image аccording to the model
        
    """

    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    data = []
    for filename in os.listdir(img_dir):
        image = transform(Image.open(img_dir + '/' + filename))
        data.append(image)
    data = torch.stack(data, 0)

    outputs = model(data.to(device).reshape(-1, 1, 28, 28))
    _, preds = torch.max(outputs, 1)

    print("Input images corresponds to character: ", map_characters[preds.cpu().numpy()])


if __name__ == "__main__":
    arg_parser = get_args_parser()
    args = arg_parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = torch.load(args.model)

    infer(model, args.img_dir)


import torch

import argparse
from tqdm import tqdm

import sign_language_mnist


def get_args_parser():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model path")
    return parser


def test(model, device, test_dataloader):
    model.eval()

    correct = 0
    correct_labels = 0
    running_predicted = 0

    for images, labels in tqdm(test_dataloader, desc="Testing", total=len(test_dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            correct_labels += labels.sum().item()
            running_predicted += preds.sum().item()

    print(f"Accuracy on test set is {correct / len(test_dataloader.dataset) * 100:.4f}%")


if __name__ == "__main__":
    arg_parser = get_args_parser()
    args = arg_parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = torch.load(args.model, map_location=device)
    test_dataloader = sign_language_mnist.get_test_loader()

    test(model, device, test_dataloader)

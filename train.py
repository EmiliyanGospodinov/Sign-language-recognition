import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler

import copy
import argparse
import os

import sign_language_mnist
import utils
from models import simple_cnn


SAVE_MODEL_DIR = "saved_models"


def get_args_parser():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    return parser


def train(model, dataloaders, criterion, optimizer, scheduler, device, writer, save=True, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_val_loss_dict = dict()
    train_val_acc_dict = dict()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        print("-" * 10)

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in training phase
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only in trianing phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.shape[0]
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

            train_val_loss_dict.update({phase: epoch_loss})
            train_val_acc_dict.update({phase: epoch_acc})

            print(f"{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            # deep copy model weights if new best model occurs
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("New best model!")

        writer.add_scalars("Loss: Train vs. Val", train_val_loss_dict, epoch)
        writer.add_scalars("Accuracy: Train vs. Val", train_val_acc_dict, epoch)
        print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    if save:
        if not os.path.exists(SAVE_MODEL_DIR):
            os.makedirs(SAVE_MODEL_DIR)
        model_path = f"{SAVE_MODEL_DIR}/{model.__class__.__name__}_best.pt"
        torch.save(model, model_path)
        print(f"Save model in {model_path}")

    return model


if __name__ == "__main__":
    arg_parser = get_args_parser()
    args = arg_parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    torch.manual_seed(0)  # ensure reproducibility
    model = simple_cnn.Net().to(device)
    dataloaders = sign_language_mnist.get_train_val_loaders()

    # Read hyperparameters from config file
    train_config = utils.get_config(args.config)["train"]
    EPOCHS = train_config["epochs"]
    LEARNING_RATE = train_config["learning_rate"]
    SAVE = train_config["save"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)  # TODO: test different optimizers

    # decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    with SummaryWriter("runs/sign_languange") as writer:
        train(model, dataloaders, criterion, optimizer, exp_lr_scheduler, device, writer, save=SAVE, num_epochs=EPOCHS)
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
from models import cnn_model

SAVE_MODEL_DIR = "saved_models"


def get_args_parser():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    return parser


def train(
    model, dataloaders, criterion, optimizer, device, writer, scheduler=None, save=True, num_epochs=25, plot=True
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # for plotting
    train_val_loss = {x: list() for x in ["train", "val"]}
    train_val_acc = {x: list() for x in ["train", "val"]}

    # for TensorBoard
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
            running_true_corrects = 0.0
            running_predicted = 0.0

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
                running_true_corrects += torch.sum(labels.data)
                running_predicted += torch.sum(preds)

            if scheduler and phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            train_val_loss[phase].append(epoch_loss)
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_rec = running_corrects / running_true_corrects
            epoch_prec = running_corrects / running_predicted
            epoch_f1 = (2 * epoch_prec * epoch_rec) / (epoch_prec + epoch_rec) 

            train_val_acc[phase].append(epoch_acc)
            writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)
            writer.add_scalar(f"Recall/{phase}", epoch_rec, epoch)
            writer.add_scalar(f"Precision/{phase}", epoch_prec, epoch)
            writer.add_scalar(f"F1-score/{phase}", epoch_f1, epoch)

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

    if plot:
        utils.plot_training(train_val_loss, train_val_acc)

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

    utils.set_random_seed(42)  # ensure reproducibility

    model = cnn_model.CNN().to(device)
    dataloaders = sign_language_mnist.get_train_val_loaders()

    # Read hyperparameters from config file
    train_config = utils.get_config(args.config)["train"]
    EPOCHS = train_config["epochs"]
    LEARNING_RATE = train_config["learning_rate"]
    SAVE = train_config["save"]
    MOMENTUM = train_config["momentum"]
    LR_GAMMA = train_config["learning_rate_gamma"]
    STEP_SIZE = train_config["learning_rate_decay_period"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)  # TODO: test different optimizers
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_GAMMA)

    with SummaryWriter("runs/sign_language") as writer:
        train(
            model,
            dataloaders,
            criterion,
            optimizer,
            device,
            writer,
            scheduler=exp_lr_scheduler,
            save=SAVE,
            num_epochs=EPOCHS,
        )

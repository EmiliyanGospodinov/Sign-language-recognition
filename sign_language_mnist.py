import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import utils


class SignLanguageMNIST(Dataset):
    def __init__(self, csv_file, phase="train", val_split=0.25, shuffle=True, transform=None, label_transform=None):
        phases = ["train", "val", "test"]
        assert phase in phases, f"Choose phase from {phases}"

        self.data = pd.read_csv(csv_file).to_numpy(np.uint8)

        train_indices, val_indices = self._train_val_split(self.data, val_split=val_split, shuffle=shuffle)
        if phase == "train":
            self.data = self.data[train_indices]
        elif phase == "val":
            self.data = self.data[val_indices]

        self.images = self.data[:, 1:].reshape(-1, 28, 28, 1)
        self.labels = self.data[:, 0]

        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        label = torch.as_tensor(label, dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.labels)

    def _train_val_split(self, data, val_split=0.25, seed=42, shuffle=True):
        indices = np.arange(len(data))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_indices = indices[: int((1 - val_split) * len(indices))]
        val_indices = indices[int((1 - val_split) * len(indices)) :]

        return train_indices, val_indices


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToPILImage(),
            # data augmentation
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
}

# Read hyperparameters from config file
train_config = utils.get_config()["train"]
TRAIN_DATASET_PATH = train_config["train_set"]["path"]
VAL_SPLIT = train_config["train_set"]["val_split"]
SHUFFLE = train_config["train_set"]["shuffle"]
TRAIN_BATCH_SIZE = train_config["batch_size"]
TRAIN_NUM_WORKERS = train_config["workers"]

test_config = utils.get_config()["test"]
TEST_DATASET_PATH = test_config["test_set"]["path"]
TEST_BATCH_SIZE = test_config["batch_size"]
TEST_NUM_WORKERS = test_config["workers"]


def get_train_val_datasets(train_dataset_path=TRAIN_DATASET_PATH):
    sign_language_datasets = {
        x: SignLanguageMNIST(
            train_dataset_path, phase=x, val_split=VAL_SPLIT, shuffle=SHUFFLE, transform=data_transforms[x]
        )
        for x in ["train", "val"]
    }
    return sign_language_datasets


def get_test_dataset(test_dataset_path=TEST_DATASET_PATH):
    return SignLanguageMNIST(test_dataset_path, phase="test", transform=data_transforms["test"])


def get_train_val_loaders():
    dataloaders = {
        x: DataLoader(
            get_train_val_datasets()[x], batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=TRAIN_NUM_WORKERS
        )
        for x in ["train", "val"]
    }
    return dataloaders


def get_test_loader():
    return DataLoader(get_test_dataset(), batch_size=TEST_BATCH_SIZE, num_workers=TEST_NUM_WORKERS)
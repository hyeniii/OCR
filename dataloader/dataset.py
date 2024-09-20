import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


class MNISTAndKaggleAZCombined(Dataset):
    """
    Class that loads MNIST data and Kaggle A-Z data combines them and binarize the labels
    """
    def __init__(self, mnist_root, az_csv_file, transforms=None):
        # Load and combine MNIST data
        mnist_train = MNIST(root=mnist_root, train=True, transform=transforms, download=True, target_transform=None)
        mnist_test = MNIST(root=mnist_root, train=False, transform=transforms, download=True, target_transform=None)
        self.mnist_data = mnist_train + mnist_test

        # Load AZ data
        self.az_data = pd.read_csv(az_csv_file).values

        # Combine MNIST and AZ data
        self.data = [(img, label) for img, label in self.mnist_data]
        self.data += [(self.az_data[i, 1:].reshape(28,28).astype('uint8'), self.az_data[i, 0] + 10.0) for i in range(len(self.az_data))]

        self.transform = transforms

        # fit labelBinarizer to create vector representation of targets
        all_labels = [label for _, label in self.data]
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(all_labels)

        self.target_classes = all_labels
        self.targets = [torch.tensor(self.label_binarizer.transform([label]).squeeze(), dtype=torch.float32)
                        for label in all_labels]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]

        if isinstance(img, np.ndarray) and self.transform:
            img = self.transform(img)
        
        one_hot_label = self.targets[idx]
        return img, one_hot_label

    def get_class_weights(self):
        classes_total = torch.stack(self.targets).sum(dim=0)
        class_weights = {}
        for i in range(len(classes_total)):
            class_weights[i] = classes_total.max() / classes_total[i]
        return class_weights
    
    def train_test_split(self, train_size=0.8):
        train_len = int(len(self.data) * 0.8)
        test_len = len(self.data) - train_len

        train_ds, test_ds = random_split(self.data, [train_len, test_len])
        return train_ds, test_ds
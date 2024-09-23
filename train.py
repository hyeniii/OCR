import os
import torch
from pathlib import Path
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from dataloader.dataset import MNISTAndKaggleAZCombined
from model.model import CustomOCRModel

DATA_PATH = Path('data/')
NUM_EPOCHS = 5
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()
LEARNING_RATE = 0.001

device = 'cude' if torch.cuda.is_available() else 'cpu'

mnist_root = 'data'
az_csv_file = 'data/A_Z Handwritten Data.csv'

data = MNISTAndKaggleAZCombined(mnist_root, az_csv_file)

# Get class weights
class_weights = data.get_class_weights()
print("Class Weights:", class_weights)

train_dataset, test_dataset = data.train_test_split()
print(len(train_dataset), len(test_dataset))

augmentation_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=28, scale=(0.95, 1.0)),
    transforms.RandomHorizontalFlip(p=0),  # No horizontal flip
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Width and height shift
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])
train_dataset.dataset.transform = augmentation_transform

# Define a simple transformation (without augmentations) for the test set
simple_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset.dataset.transform = simple_transform

train_dataloader = DataLoader(train_dataset,
                                batch_size = BATCH_SIZE,
                                shuffle = True,
                                num_workers = NUM_WORKERS,
                                pin_memory = True) #  lets DataLoader allocate samples in page-locked memory, which speeds-up the transfer from CPU dataloading to GPU training
test_dataloader = DataLoader(test_dataset,
                                batch_size = BATCH_SIZE,
                                shuffle = False,
                                num_workers = NUM_WORKERS,
                                pin_memory = True)

name_labels = '0123456789'
name_labels += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
name_labels = [l for l in name_labels]

model = CustomOCRModel(input_shape=1, output_shape=len(name_labels))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

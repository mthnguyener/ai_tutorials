#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Self-Supervised Learning Module

"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Data Augmentation
class Cutout:
    """
    Initializes a new instance of the class with the given number of holes
        and length.

    :param n_holes: An integer representing the number of holes.
    :param length: An integer representing the length.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(max(0, y - self.length // 2))
            y2 = int(min(h, y + self.length // 2))
            x1 = int(max(0, x - self.length // 2))
            x2 = int(min(w, x + self.length // 2))

            mask[y1:y2, x1:x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)

        return img * mask

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    Cutout(n_holes=1, length=16)  # Introduce holes in images
])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CIFAR-10 Train Dataset Test Dataset
batch_size = 128
num_workers = 16

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                 transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)


# Model Architecture
class Encoder(nn.Module):
    """
    Initializes the Encoder object by creating a sequential neural network
        with convolutional layers.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        return self.encoder(x)


class ProjectionHead(nn.Module):
    """
    Initializes the ProjectionHead object with the specified input, hidden,
        and output dimensions.

    :param input_dim: The input dimension of the projection head.
    :param hidden_dim: The hidden dimension of the projection head.
    :param output_dim: The output dimension of the projection head.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.projection_head(x)


class SimCLR(nn.Module):
    """
    Initializes the SimCLR class.

    :param encoder: The encoder object.
    :param projection_head: The projection head object.
    """

    def __init__(self, encoder, projection_head):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten the features
        projections = self.projection_head(features)
        return features, projections


class ContrastiveLoss(nn.Module):
    """
    Initializes the ContrastiveLoss object with the given temperature.

    :param temperature: The temperature for the ContrastiveLoss.
    """

    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        bs = features.size(0)
        features = nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features,
                                         features.T) / self.temperature
        loss = F.cross_entropy(similarity_matrix, torch.arange(bs).cuda())
        return loss


def train_simclr(learning_rate: float = 0.0005, num_epochs: int = 100):
    """
    Train a SimCLR model.

    :param learning_rate: Learning rate for the optimizer (default: 0.0005).
    :param num_epochs: The number of epochs for training (default: 100).

    :return: The trained SimCLR model.
    """
    print("\nStarted SimCLR training...")
    start = time.time()

    # Compiling the model
    encoder = Encoder().to(device)

    # Update projection head input dimension
    projection_head = ProjectionHead(2048, 256, 128).to(device)
    model = SimCLR(encoder, projection_head).to(device)
    model.train()  # Set model to training mode

    # Define optimizer and loss function
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=num_epochs,
                                                           eta_min=0)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            images, _ = batch
            images = images.to(device)
            features, projections = model(images)
            loss = criterion(features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Update learning rate
        scheduler.step()

        # Print information every 5 epochs or at the last epoch
        if (epoch + 1) % (num_epochs/20) == 0 \
                or epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Loss: {total_loss / len(train_loader):.4f}")

    end = time.time()
    print(f"Training took {(end - start) / 60} mins")

    return model


# ----- Classifier -----
# Define a simple linear classifier
class LinearClassifier(nn.Module):
    """
    Constructor for the LinearClassifier class.

    :param input_dim: Dimension of the input features.
    :param num_classes: Number of classes for classification.
    """

    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Increased hidden layer size
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.2)  # Dropout layer for regularization
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(256, num_classes)  # Additional hidden layer

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_classifier(learning_rate: float = 0.0001,
                     model=None,
                     num_epochs: int = 100):
    """
    Train a linear classifier.

    :param learning_rate: Learning rate for the optimizer (Default: 0.0001).
    :param model: The model to train the classifier on.
    :param num_epochs: The number of epochs to train (Default: 100).

    :return: The trained linear classifier model.
    """
    print("\nStarted classifier training...")
    start = time.time()

    model.eval()
    model.to(device)

    # Initialize the classifier (assuming reduced feature dim is 192 * 4 * 4)
    classifier = LinearClassifier(input_dim=192 * 4 * 4,
                                  num_classes=10).to(device)
    classifier.train()

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(),
                                 lr=learning_rate,
                                 weight_decay=0.001)

    # Train the linear classifier
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            # Reshape features if necessary (same as in extract_features)
            features = features.view(features.size(0), -1).to(device)
            # print(f"Feature Shape: {features.shape}")
            labels = labels.to(device)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print information every 5 epochs or at the last epoch
        if (epoch + 1) % (num_epochs/20) == 0 \
                or epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    end = time.time()
    print(f"Training took {(end - start) / 60} mins")

    return classifier


def eval_classifier(classifier=None):
    """
    Evaluate the classifier on the test set to calculate the accuracy.

    :param classifier: The classifier model to evaluate (default:None).
    """
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nAccuracy on the test set: {(100 * correct / total):.2f}%")


if __name__ == '__main__':
    simclr = train_simclr(num_epochs=50)
    image_classifier = train_classifier(model=simclr, num_epochs=10)
    eval_classifier(classifier=image_classifier)

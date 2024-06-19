import os
import numpy as np
import pandas as pd
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import opendatasets as od
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modelCNN import SimpleCNN, CNN2
from data_loader import train_loader, val_loader

print("imported")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path, name):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Training Loss: {train_losses[-1]:.4f}, '
              f'Validation Loss: {val_losses[-1]:.4f}, '
              f'Accuracy: {accuracy:.2f}%')

    # Save the model
    torch.save(model.state_dict(), save_path)
    print('Model saved to' + save_path)

    # Plot and save the loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(name + '.png')
    
print("loaded")
criterion = nn.CrossEntropyLoss()

### Training simple CNN model:

modelCNN1 = SimpleCNN()
optimizerCNN1 = optim.Adam(modelCNN1.parameters(), lr=0.001)
train_model(modelCNN1, train_loader, val_loader, criterion, optimizerCNN1, num_epochs=20, save_path = 'simple_cnn.pth', name = 'simple_cnn')

### Training CNN2 model:
modelCNN2 = CNN2()
optimizerCNN2 = optim.Adam(modelCNN2.parameters(), lr=0.001)

train_model(modelCNN2, train_loader, val_loader, criterion, optimizerCNN2, num_epochs=20, save_path = 'cnn2.pth', name = 'cnn2')





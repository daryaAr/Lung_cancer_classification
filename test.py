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
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modelCNN import SimpleCNN, CNN2
from torchvision.models import resnet50, ResNet50_Weights
from data_loader import test_loader

with open('lung_cancer_data.pkl', 'rb') as f:
    data = pickle.load(f)

categories = ['lung_aca', 'lung_n', 'lung_scc']



def evaluate_model_CNN(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    losses = []
    predictions = []
    ground_truths = []
    images_to_save = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
            
            if len(images_to_save) < 10:
                images_to_save.extend(images.cpu().numpy())
    
    accuracy = 100 * correct / total
    average_loss = test_loss / len(test_loader)

    print(f'Accuracy: {accuracy}%')
    print(f'Test Loss: {average_loss:.4f}')
    
    return losses, predictions, ground_truths, images_to_save



# Load the saved model state
criterion = nn.CrossEntropyLoss()

model1 = SimpleCNN()
model1.load_state_dict(torch.load('simple_cnn.pth'))
model1.eval() 
losses1, predictions1, ground_truths1, images1 = evaluate_model_CNN(model1, test_loader, criterion)

model2 = CNN2()
model2.load_state_dict(torch.load('cnn2.pth'))
model2.eval() 
losses2, predictions2, ground_truths2, images2 = evaluate_model_CNN(model2, test_loader, criterion)

model3 = resnet50(weights=ResNet50_Weights.DEFAULT)
model3.fc = nn.Linear(model3.fc.in_features, 3)
model3.load_state_dict(torch.load('resnet50.pth'))
model3.eval() 
losses3, predictions3, ground_truths3, images3 = evaluate_model_CNN(model3, test_loader, criterion)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(losses1, label='SimpleCNN')
plt.plot(losses2, label='CNN2')
plt.plot(losses3, label='ResNet50')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()
plt.savefig('loss_comparison.png')
plt.show()

# Display some images with predictions and ground truths
def show_predictions(images, predictions, ground_truths, filename):
    fig = plt.figure(figsize=(15, 10))
    for i in range(min(10, len(images))):
        ax = fig.add_subplot(2, 5, i + 1)
        img = images[i].transpose(1, 2, 0)  # Change to HWC format
        img = (img * 255).astype(np.uint8)  # Rescale to 0-255
        ax.imshow(img)
        ax.set_title(f'Pred: {predictions[i]}, GT: {ground_truths[i]}')
        ax.axis('off')
    plt.savefig(filename)
    plt.show()


show_predictions(images1, predictions1, ground_truths1, 'predictions_simple_cnn.png')
show_predictions(images2, predictions2, ground_truths2, 'predictions_cnn2.png')
show_predictions(images3, predictions3, ground_truths3, 'predictions_resnet50.png')
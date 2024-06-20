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

dataset = "https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images"
od.download(dataset)

data_dir = "./lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets"
categories = ['lung_aca', 'lung_n', 'lung_scc']

def load_images_from_folder(folder, label):
    images = []
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return images
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        print(f"Processing file: {img_path}")
        img = Image.open(img_path)
        img = img.resize((128, 128))  
        img = np.array(img)
        img = img / 255.0  # Normalization
        if img is not None:
            images.append((img, label))
    return images

data = []
for category in categories:
    folder = os.path.join(data_dir, category)
    label = categories.index(category)
    print(f"Loading images from folder: {folder} with label: {label}")
    data.extend(load_images_from_folder(folder, label))

print(f"Total images loaded: {len(data)}")


with open('lung_cancer_data.pkl', 'wb') as f:
    pickle.dump(data, f)

categories = ['lung_aca', 'lung_n', 'lung_scc']
    
# Function to save sample images
def save_samples(data, categories, num_samples=5, filename='sample_images.png'):
    fig, axes = plt.subplots(len(categories), num_samples, figsize=(15, 10))
    fig.suptitle('Sample Images from Each Category', fontsize=16)

    for i, category in enumerate(categories):
        category_data = [img for img, label in data if label == i]
        for j in range(num_samples):
            ax = axes[i, j]
            image = category_data[j]
            ax.imshow(image)
            ax.set_title(categories[i])
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    plt.savefig(filename)
    plt.close()

save_samples(data, categories, num_samples=5, filename='sample_images.png')


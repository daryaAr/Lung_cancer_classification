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

with open('lung_cancer_data.pkl', 'rb') as f:
    data = pickle.load(f)

"""
This code defines a series of image preprocessing functions and a transformation pipeline for preparing
histopathology images for machine learning models. It includes the following components:

1. **Histogram Equalization Function**:
    - `histogram_equalization(image)`: Enhances the global contrast of an image by equalizing the histogram 
      of the Y (luminance) channel in the YUV color space.

2. **CLAHE Function**:
    - `clahe(image)`: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the Y (luminance) 
      channel in the YUV color space to enhance local contrast and highlight details.

3. **Random Morphological Transformation Class**:
    - `RandomMorphologicalTransform`: Applies a random morphological operation (either dilation or erosion) 
      to the image to introduce variability and augment the dataset.

4. **Transformation Pipeline**:
    - `transform = transforms.Compose([...])`: A pipeline of transformations that includes resizing, histogram 
      equalization, CLAHE, random morphological transformations, random rotations, random crops, and flips, 
      tensor conversion, and normalization. This pipeline prepares the images for training robust machine 
      learning models by augmenting and normalizing them.

5. **Custom Dataset Class**:
    - `HistopathologyDataset`: A custom dataset class that loads histopathology images and applies the defined 
      transformation pipeline. This class is compatible with PyTorch's DataLoader for efficient data loading 
      during model training.

The combined effect of these components is to preprocess and augment histopathology images, enhancing contrast,
introducing random variability, and normalizing the images to improve the performance and generalization 
capability of machine learning models.
"""

def histogram_equalization(image):
    image_yuv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_equalized = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(image_equalized)

def clahe(image):
    image_yuv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])
    image_clahe = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(image_clahe)

# Define custom transformations
class RandomMorphologicalTransform:
    def __call__(self, image):
        image = np.array(image)
        if random.choice([True, False]):
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)
        else:
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
        return Image.fromarray(image)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(histogram_equalization),
    transforms.Lambda(clahe),
    transforms.Lambda(RandomMorphologicalTransform()),
    transforms.RandomRotation(degrees=10),  
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.9, 1.0)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class HistopathologyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))  
        if self.transform:
            image = self.transform(image)
        return image, label

"""
Below code defines a `DataLoaderSplitter` class which is used to split a given dataset into training,
validation, and test sets, and then creates DataLoader objects for each of these sets. The class
initializes with the dataset and the desired batch size, and provides a method `split_data` to 
perform the data splitting. The splits are performed with 70% of the data for training, 15% for validation,
and 15% for testing.

Usage:
1. Initialize the `HistopathologyDataset` with the data and transformation pipeline.
2. Create an instance of `DataLoaderSplitter` with the dataset and the desired batch size.
3. Call the `split_data` method to split the dataset and create DataLoader objects for training, 
   validation, and testing.
"""
class DataLoaderSplitter:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def split_data(self):
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.15 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


histopathology_dataset = HistopathologyDataset(data, transform)
splitter = DataLoaderSplitter(histopathology_dataset, batch_size=64)

splitter.split_data()

train_loader = splitter.train_loader
val_loader = splitter.val_loader
test_loader = splitter.test_loader 
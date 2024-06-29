# Few Shot Learning -Chiral Molecule Dataset
by Megan Tran

## Table of Contents
* [Summary of Results](#Summary-of-Results)
* [Purpose of Program](#Purpose-of-program)
* [Technologies](#technologies)
* [Setup](#setup)
* [Parameters](#Parameters)
* [Credits](#Credits)

## Summary of Results

![image](https://github.com/MeganTran6023/ChiralMolecule_TfClassification/assets/68253811/5cc870e6-40ab-4019-af0e-4b9b22d9cc4c)

**Analysis:**

The model was able to learn the training dataset well as there was 100% accuracy on the 2nd epoch. This makes sense due to the low learning rate and small training dataset of 5 items. However, this was not condusive for evaluating the model's performance as it's accuracy for the validation dataset (ds) significantly decreased from 60% to 20%. Furthermore, the validation loss increased with the number of epochs.

To resolve this, I could increase the number of items in the dataset to then do a 80:20 split train:test on it for proper classification.

## Purpose of Program

This program was created to run a few shot deep learning model on my own collected dataset of 10 images or chiral/achiral molecules. The flowchart below is a general overview of how this project was completed

![image](https://github.com/MeganTran6023/ChiralMolecule_TfClassification/assets/68253811/153eb67f-a963-48a6-b1b3-28a53b039b45)

## Technologies

* Jupyter Notebook

* Python3

## Setup

Import necessary packages and modules:

```
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd

# Model Execution
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

## This is the model used:
from tensorflow.keras.applications import InceptionV3

```

## Parameters

Batch_size -> how many items to bunch together when running it through model for training

    * I chose 1 to test how accurate the model would predict images labels since this learns images one by one. However, this utilizes a high level of computational cost.

For the base_model, I set the weights to be 'imagenet' so it would use information learned from the imagenet dataset and apply it to my custom dataset.

I also set the number of epochs to be 10 since i already had set a small value for the learning rate as well as have only 5 items for the model to learn from (out of 10).

Train:test split was 5:5 to see how it would perform given the small dataset.

I also had to change y-variable classes to binary to run the epochs:
```
#assign numbers to labels classificaiton
molecule_class_num = df_molecules['label'].nunique()
```
## Credits

Code referenced: [ZALCODE](https://www.kaggle.com/code/zalcode/super-simple-butterfly-image-classification#Project-Description)


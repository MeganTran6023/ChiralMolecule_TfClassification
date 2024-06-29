# Few Shot Learning -Chiral Molecule Dataset
by Megan Tran

## Table of Contents
* [Summary of Results](#Summary-of-Results)
* [Purpose of Program](#Purpose-of-program)
* [Technologies](#technologies)
* [Setup](#setup)
* [Using the Program](#Using-the-Program)
* [Credits](#Credits)

## Summary of Results

![image](https://github.com/MeganTran6023/ChiralMolecule_TfClassification/assets/68253811/5cc870e6-40ab-4019-af0e-4b9b22d9cc4c)

**Analysis:**

The model was able to learn the training dataset well as there was 100% accuracy on the 2nd epoch. This makes sense due to the low learning rate and small training dataset of 5 items. However, this was not condusive for evaluating the model's performance as it's accuracy for the validation dataset (ds) significantly decreased from 60% to 20%. Furthermore, the validation loss increased with the number of epochs.

To resolve this, I could increase the number of items in the dataset to then do a 80:20 split train:test on it for proper classification.

## Purpose of Program

This program was created to run a few shot deep learning model on my own collected dataset of 10 images or chiral/achiral molecules. The flowchart below is a general overview of how this project was completed

![image](https://github.com/MeganTran6023/ChiralMolecule_TfClassification/assets/68253811/153eb67f-a963-48a6-b1b3-28a53b039b45)

## Technologies Used

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

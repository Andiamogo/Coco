import matplotlib
matplotlib.use('Agg')
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from VGGNet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

EPOCHS = 100
INIT_LR = 0.001
BS = 32
IMAGE_DIMENSIONS = (96, 96, 3)

data = []
labels = []

print('[INFO] Image loading...')
image_paths = sorted(list(paths.list_images('C:\\Users\\maxim\\Python\\Coco\\data')))
random.seed(42)
random.shuffle(image_paths)
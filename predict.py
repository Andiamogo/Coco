from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import random
import PIL
from PIL import ImageColor
import math


def load_cnn():
    print('[INFO] CNN Network loading...')
    model = load_model('./models/model.h5')
    lb = pickle.loads(open('lab.pickle', 'rb').read())

    return model, lb

def classify(filename, model, lb):
    print(filename)
    image = cv2.imread(filename)
    output = image.copy()

    image = cv2.resize(image, (96, 96))
    image = image.astype('float') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    
    return label, proba[idx] * 100

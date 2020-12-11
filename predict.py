from keras.preprocessing.image import img_to_array
from keras.models import load_model
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


total = {
    "1c": 0,
    "2c": 0,
    "5c": 0,
    "10c": 0,
    "20c": 0,
    "50c": 0,
    "1e": 0,
    "2e": 0,
}

succeded = {
    "1c": 0,
    "2c": 0,
    "5c": 0,
    "10c": 0,
    "20c": 0,
    "50c": 0,
    "1e": 0,
    "2e": 0,
}

model, lb = load_cnn()
for root, dirs, files in os.walk('./data'):
    for file in files:
        name, score = classify(
            os.path.join(root, file),
            model,
            lb
        )
        if name == root.split('/')[-1]:
            succeded[root.split('/')[-1]] += 1
        total[root.split('/')[-1]] += 1

for e in succeded:
    print(f'Correct prediction rate for {e} is : {succeded[e]/total[e]}')

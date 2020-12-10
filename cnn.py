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
image_paths = sorted(list(paths.list_images('./data')))
random.seed(42)
random.shuffle(image_paths)

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0]))
    image = img_to_array(image)
    #data
    data.append(image)
    #labels
    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype='float') / 255.
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# splitting dataset
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

model = SmallerVGGNet.build(width=IMAGE_DIMENSIONS[1], height=IMAGE_DIMENSIONS[0], depth=IMAGE_DIMENSIONS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit_generator(
    datagen.flow(train_x, train_y, batch_size=BS),
    validation_data=(test_x, test_y),
    steps_per_epoch=len(train_x) // BS,
    epochs=EPOCHS,
    verbose=1
)

model.save('models/model.h5')

f = open('lab.pickle', 'wb')
f.write(pickle.dumps(lb))
f.close()
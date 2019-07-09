import plaidml.keras
plaidml.keras.install_backend()

from scipy import ndimage
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import numpy as np
import csv
import cv2

lines = []
with open('./training-data/track-1-and-2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print('Read {} lines'.format(len(lines)))

images = []
measurements = []
correction_factor = 0.2
for line in lines:
    for i in range(3):
        source_path = line[i]
        image = ndimage.imread(source_path)
        images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction_factor)
    measurements.append(measurement - correction_factor)

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(-measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)),
    Cropping2D(cropping=((55,25), (0,0))),
    Conv2D(filters=6, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(120),
    Dropout(0.3),
    Dense(84),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')

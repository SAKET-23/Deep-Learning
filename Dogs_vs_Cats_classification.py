# loading the data set

import numpy as np
import os
import cv2
import matplotlib.pyplot as pyplot
import random


IMG_SIZE = 50


DATADIR = "C:/Users/SAKET/Desktop/ML WITH PYTHON/github/PetImages"
Categories = ["Dog","Cat"]


training_data = []
for category in Categories:

        path = os.path.join(DATADIR,category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE )
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])

            except  Exception as e:
                pass


random.shuffle(training_data)

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1 ,IMG_SIZE,IMG_SIZE,1)
y = np.array(y).reshape(-1 ,1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,Activation,Dropout

X = tf.keras.utils.normalize(X , axis = 1)

model = Sequential()

model.add(Conv2D(128,(3,3) , input_shape = X.shape[1:]))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten())
model.add(Dense(500 , activation="relu"))
model.add(Dropout(0.5))


model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss = "binary_crossentropy",optimizer="adam",metrics = ["accuracy"])

model.fit(X, y, batch_size = 64, validation_split = 0.1, epochs = 10, shuffle=True )

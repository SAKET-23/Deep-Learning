
# Loading the dataset

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# normalizing the dataset
x_train = tf.keras.utils.normalize(x_train , axis=1)
x_test  = tf.keras.utils.normalize(x_test, axis=1)

# training the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten

model = Sequential()
model.add(Flatten())

model.add(Dense(256 , activation= "relu"))
model.add(Dense(128 , activation= "relu"))

model.add(Dense(10 , activation= tf.nn.softmax))
model.compile(optimizer = "adam" , loss = "sparse_categorical_crossentropy" , metrics = ["accuracy"])

model.fit(x = x_train,y = y_train , batch_size=32 , epochs= 100 , validation_data=(x_test,y_test))

# Prediction using model
pred = model.predict(x_test)

# evaluation of predictions

val_loss , val_acc = model.evaluate(x_test,y_test)
print(val_loss ," ",val_acc)

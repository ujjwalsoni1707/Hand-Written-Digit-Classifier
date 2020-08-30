# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

#len(X_train)
#len(X_test)
#X_train[0].shape

#plt.matshow(X_train[0])
#print(y_train[0])

#Scalling the data
X_train = X_train / 255
X_test = X_test / 255
#flattning the data
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

#X_train_flattened.shape

#Nueral Network with No Hidden Layer
'''model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
print(model.evaluate(X_test_flattened, y_test))
y_predicted = model.predict(X_test_flattened)
print(y_predicted[0])
plt.matshow(X_test[0])
print(np.argmax(y_predicted[0]))

#Confusion Matrix
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
'''

#Neural Network with one hidden layer
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
print(model.evaluate(X_test_flattened,y_test))
y_predicted = model.predict(X_test_flattened)
print(y_predicted[0])
plt.matshow(X_test[0])
print(np.argmax(y_predicted[0]))
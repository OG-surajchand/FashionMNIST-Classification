# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:17:43 2021

@author: sctha
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

df = tf.keras.datasets.fashion_mnist
(train_image, train_label),(test_image, test_label) = df.load_data()

class_dress = ['Top','Pants','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

plt.figure()
plt.imshow(train_image[0])
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()

train_image = train_image/255.0
test_image = test_image/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_image,train_label,epochs=10)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_model,epochs=10)
accuracy = cross_val_score(estimator=model, X= train_image, y= train_label,cv = 10)
mean = accuracy.mean()
 
predicted_image = model.predict(test_image)

#Sample Predictions

print("Predicted class: " +  class_dress[np.argmax(predicted_image[6])])
print("Actual Test class: " +  class_dress[test_label[6]])

Results = []

for i in range(0,10000):
    Results.append("Predicted: " + class_dress[np.argmax(predicted_image[i])]+" -- "+ "Actual: " + class_dress[test_label[i]])



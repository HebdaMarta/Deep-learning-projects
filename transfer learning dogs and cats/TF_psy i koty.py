import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import layers
import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns

import glob
import numpy as np
import os
import pandas as pd


def show_images(x, y, idx):
  image = x[idx]
  plt.figure(figsize=(4,2))
  plt.imshow(image)
  if np.argmax(y[idx]) == 0:
      kind = "cat"
  else:
      kind = "dog"
  plt.title("This is a {}".format(kind))
  plt.show()


def train_data(label, data_dir):
    i = 0
    for img in os.listdir(data_dir):
        if i < 2000:
          path = os.path.join(data_dir, img)
          img = cv2.imread(path, 1)

          img = cv2.resize(img, (imgsize, imgsize))
          x.append(np.array(img))
          y.append(str(label))
          i= i+1
        else:
          break


# Data
x = [] 
y = []
imgsize = 299
path_dogs = "/content/drive/MyDrive/dogs"
path_cats = "/content/drive/MyDrive/cats"

train_data('Dog', path_dogs)
train_data('Cat', path_cats)

x = np.array(x)
x = x/255

label_encoder=LabelEncoder()
y = label_encoder.fit_transform(y)
y = OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()
y = y.astype(np.float32)

print("y.shape: ", y.shape)
print("x.shape: ", x.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state =0)

print("y.shape: ", y_train.shape)
print("x.shape: ", x_train.shape)

# Display image
classes = {0: 'cats', 1: 'dogs'}
show_images(x_train, y_train, 12)


# Model
input_shape = x_train.shape[1:]
inputLayer = tf.keras.Input(shape=(input_shape))

base_model = tf.keras.applications.InceptionV3(
    weights='imagenet',
    input_shape=input_shape,
    include_top=False)

base_model.trainable = False

output = base_model.layers[-1].output
x = layers.GlobalAveragePooling2D()(output)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout (0.2)(x)
x = layers.Dense(2, activation="sigmoid")(x)

Inception_model = tf.keras.Model(base_model.input, x)

print(Inception_model.summary())

Inception_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

# Training
history = Inception_model.fit(x_train, y_train, batch_size=100, epochs=10)

# Testing
y_pred = Inception_model.predict(x_test)
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), output_dict=True, digits=4)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, fmt='.4f')
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), digits=4))
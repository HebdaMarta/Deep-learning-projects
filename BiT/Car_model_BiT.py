import tensorflow as tf
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import cv2
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers
import datetime
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator

sedan = '/content/drive/MyDrive/BAZY DANYCH/RODZAJE SAMOCHODOW/data/Sedan'
minivan = '/content/drive/MyDrive/BAZY DANYCH/RODZAJE SAMOCHODOW/data/Minivan'
limousine = '/content/drive/MyDrive/BAZY DANYCH/RODZAJE SAMOCHODOW/data/Limousine'
hatchback = '/content/drive/MyDrive/BAZY DANYCH/RODZAJE SAMOCHODOW/data/Hatchback'
coupe = '/content/drive/MyDrive/BAZY DANYCH/RODZAJE SAMOCHODOW/data/Coupe'
convertible = '/content/drive/MyDrive/BAZY DANYCH/RODZAJE SAMOCHODOW/data/Convertible'
buggy = '/content/drive/MyDrive/BAZY DANYCH/RODZAJE SAMOCHODOW/data/Buggy'

X = []
y_label = []
imgsize = 256


def train_data(label, data_dir):
    for img in os.listdir(data_dir):
        path = os.path.join(data_dir, img)
        img = cv2.imread(path, 1)
        if img is None:
            continue
        img = cv2.resize(img, (imgsize, imgsize))
        X.append(np.array(img))
        y_label.append(str(label))


paths = [sedan, minivan, limousine, hatchback, coupe, convertible, buggy]
categories = ['sedan', 'minivan', 'limousine', 'hatchback', 'coupe', 'convertible', 'buggy']

for indeks, path in enumerate(paths):
    train_data(categories[indeks], path)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label)
X = np.array(X)

y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

y = y.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("y_train.shape: ", y_train.shape)
print("y_test.shape: ", y_test.shape)
print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)

SCHEDULE_LENGTH = 500
BATCH_SIZE = 256
STEPS_PER_EPOCH = 2672 // BATCH_SIZE
SCHEDULE_BOUNDARIES = [200, 300, 400]

SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE

# def preprocessing_functions(image):
#     seed = (1, 2)
#     image = tf.image.stateless_random_crop(image, size=[240, 240, 3], seed=seed)
#     image = np.array(image)
#     image = cv2.resize(image, (imgsize, imgsize))
#     return image

datagen_train = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True)

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)

datagen_train.fit(X_train)
datagen_test.fit(X_test)

lr = 0.003 * BATCH_SIZE / 512
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES,
                                                                   values=[lr, lr * 0.1, lr * 0.001, lr * 0.0001])

model = hub.load("https://tfhub.dev/google/bit/m-r101x1/1")
module = hub.KerasLayer(model)

inputs = tf.keras.layers.Input(shape=(X_test.shape[1:]))
logits = module(inputs)
logits = tf.keras.layers.Dense(1024, activation='relu')(logits)
logits = tf.keras.layers.Dropout(0.2)(logits)
logits = tf.keras.layers.Dense(512, activation='relu')(logits)
logits = tf.keras.layers.Dropout(0.2)(logits)
logits = tf.keras.layers.Dense(256, activation='relu')(logits)
logits = tf.keras.layers.Dropout(0.1)(logits)
logits = tf.keras.layers.Dense(128, activation='relu')(logits)
logits = tf.keras.layers.Dropout(0.1)(logits)
outputs = tf.keras.layers.Dense(7, activation='softmax', dtype='float32')(logits)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01), metrics=["accuracy"])

print(model.summary())

history = model.fit(
    datagen_train.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=50,
    validation_data=datagen_test.flow(X_test, y_test)
)


fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(20,10))

ax[0][0].plot(history.history["loss"])
ax[0][1].plot(history.history["accuracy"])
ax[1][0].plot(history.history["val_loss"])
ax[1][1].plot(history.history["val_accuracy"])

ax[0][0].set_title("train loss")
ax[0][1].set_title("train accuracy")
ax[1][0].set_title("validation loss")
ax[1][1].set_title("validation accuracy")
plt.show()


y_pred = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), digits=4))
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
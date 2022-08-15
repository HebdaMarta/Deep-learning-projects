import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from sklearn.preprocessing import OneHotEncoder
from keras import layers, activations
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import regularizers
import cv2
import tensorflow as tf
from tensorflow.keras import layers, activations


def set_value_lr(epoch):
    lr = 0.001
    if epoch > 75:
        lr = 0.0005
    if epoch > 100:
        lr = 0.0003
    if epoch > 150:
        lr = 0.0002
    if epoch > 200:
        lr = 0.0001
    return lr

# plot charts
def plot_charts(history):
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


# Data
cifar = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

y_train = OneHotEncoder().fit_transform(y_train.reshape(-1,1)).toarray()
y_train = y_train.astype(np.float32)

y_test = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()
y_test = y_test.astype(np.float32)

print("x_test.shape: ", x_test.shape)
print("x_train.shape: ", x_train.shape)

print("y_test.shape: ", y_test.shape)
print("y_train.shape: ", y_train.shape)


# Display a sample photo
n = 240
plt.matshow(x_train[n])
plt.show()

# Augmentation
aug_train = ImageDataGenerator(
  featurewise_center=True,
  featurewise_std_normalization=True,
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=True,
	vertical_flip=False,
	fill_mode="nearest")

aug_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

aug_train.fit(x_train)
aug_test.fit(x_test)


# Create model

def residual_block(x, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = x
    f1, f2, f3 = nb_channels

    x = layers.Conv2D(f1, (1,1), padding='valid', strides=_strides, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    x = layers.Conv2D(f2, (3,3), padding='same', strides=_strides, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    x = layers.Conv2D(f3, (1,1), padding='valid', strides=_strides, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(f3, kernel_size=(1, 1), strides=_strides, padding='valid')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([shortcut, x])
    x = layers.Activation(activations.relu)(x)
    return x

weight_decay = 1e-4

input = layers.Input(shape=(x_test.shape[1:]))

x = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(input)
x = layers.BatchNormalization()(x)
x = layers.Activation(activations.relu)(x)
x = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(activations.relu)(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.3)(x)

x = residual_block(x,[256, 256, 256],(1,1),True)

x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.3)(x)

x = residual_block(x,[512,512, 512],(1,1),True)

x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.4)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(10, name="wyjscie", activation='softmax')(x)

model = tf.keras.Model(input, output)

print(model.summary())

# Model compilation
model.compile(optimizer=tf.keras.optimizers.Adam(0.001,decay=1e-6),loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])


# Train
history = model.fit(aug_train.flow(x_train, y_train, batch_size=128),
                    validation_data= aug_test.flow(x_test, y_test, batch_size=128), 
                    epochs=200, 
                    callbacks=[LearningRateScheduler(set_value_lr)])

# Test
y_pred = model.predict(x_test)
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), output_dict=True, digits=4)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, fmt='.4f')
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), digits=4))

# plot charts
plot_charts(history)
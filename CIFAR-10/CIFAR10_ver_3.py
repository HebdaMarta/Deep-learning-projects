import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from keras import layers, activations
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import regularizers
import tensorflow as tf
from tensorflow.keras import layers, activations


def cosine_decay(step):
    step = min(step, epochs)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / epochs))
    return initial_learning_rate * cosine_decay


# plot charts
def plot_charts(history):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    ax[0][0].plot(history.history["loss"])
    ax[0][1].plot(history.history["accuracy"])
    ax[1][0].plot(history.history["val_loss"])
    ax[1][1].plot(history.history["val_accuracy"])

    ax[0][0].set_title("train loss")
    ax[0][1].set_title("train accuracy")
    ax[1][0].set_title("validation loss")
    ax[1][1].set_title("validation accuracy")
    plt.show()


def resnet_layer(input, shortcut, neuron, kernel_size=(3, 3)):
    x = layers.Conv2D(neuron, kernel_size, strides=(1 if not shortcut else 2), padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(neuron, kernel_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(
        x)

    if shortcut:
        input = layers.Conv2D(neuron, (1, 1), strides=2, padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay))(input)
        input = layers.BatchNormalization()(input)

    x = layers.Add()([input, x])
    x = layers.ReLU()(x)

    return x

# Data
cifar = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

y_train = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).toarray()
y_train = y_train.astype(np.float32)

y_test = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
y_test = y_test.astype(np.float32)

print("x_test.shape: ", x_test.shape)
print("x_train.shape: ", x_train.shape)

print("y_test.shape: ", y_test.shape)
print("y_train.shape: ", y_train.shape)

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
initial_learning_rate = 0.1
epochs = 230
weight_decay = 1e-4
neurons = [64, 128, 256, 512]

input = layers.Input(shape=(x_test.shape[1:]))
x = layers.Conv2D(64, (5, 5), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(input)
x = layers.BatchNormalization()(x)
x = layers.Activation(activations.relu)(x)

for number in range(4):
    for block in range(3):
        x = resnet_layer(x, shortcut=(block == 0 and number != 0), neuron=neurons[number])

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(1024, activation='relu')(x)
output = layers.Dense(10, name="wyjscie", activation='softmax')(x)

model = tf.keras.Model(input, output)

print(model.summary())

# Model compilation
model.compile(optimizer=tf.keras.optimizers.Adam(initial_learning_rate, decay=1e-6),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

# Train
history = model.fit(aug_train.flow(x_train, y_train, batch_size=128),
                    validation_data=aug_test.flow(x_test, y_test, batch_size=128),
                    epochs=epochs,
                    callbacks=[LearningRateScheduler(cosine_decay)])

# plot charts
plot_charts(history)

# Test
y_pred = model.predict(aug_test.flow(x_test))
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), output_dict=True, digits=4)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, fmt='.4f')
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), digits=4))


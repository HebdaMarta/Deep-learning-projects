
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import regularizers

def set_value_lr(epoch):
    lr = 0.001
    if epoch > 75:
        lr = 0.0005
    if epoch > 100:
        lr = 0.0003
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
x_train, x_test = x_train / 255.0, x_test / 255.0

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
n = 580
plt.matshow(x_train[n])
plt.show()

# Augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=True,
	vertical_flip=False,
	fill_mode="nearest")

# Create model

model = tf.keras.Sequential()
model.add(layers.InputLayer(input_shape=x_test.shape[1:]))

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.2))
 
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))
 
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.4))
 
model.add(layers.Flatten())
model.add(layers.Dense(10, name="wyjscie", activation='softmax'))

print(model.summary())

# Model compilation
model.compile(optimizer=tf.keras.optimizers.Adam(0.001,decay=1e-6),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])


# Train
history = model.fit(aug.flow(x_train, y_train, batch_size=128), 
                    validation_data= (x_test, y_test), 
                    epochs=200, 
                    callbacks=[LearningRateScheduler(set_value_lr)]
                    )

# Test
y_pred = model.predict(x_test)
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), output_dict=True, digits=4)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, fmt='.4f')
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), digits=4))

# plot charts
plot_charts(history)
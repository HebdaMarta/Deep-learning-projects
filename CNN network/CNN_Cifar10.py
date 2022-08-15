import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


def create_cnn_model(_input_shape, _conv_num, _dense_num, _act, _output_num):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=_input_shape))

    for idx, one_conv_filters in enumerate(_conv_num):
        model.add(
            layers.Conv2D(
                filters=one_conv_filters,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=_act,
                name="konwolucja_" + str(idx)
            )
        )

        model.add(
            layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding="SAME",
                name="redukcja_" + str(idx)
            )
        )

        model.add(
            layers.Dropout(0.5)
        )

    model.add(layers.Flatten(name="splaszczenie"))

    for idx, one_dense_units in enumerate(_dense_num):
        model.add(
            layers.Dense(
                units=one_dense_units,
                activation=_act,
                name="gesta_" + str(idx)
            )
        )

        model.add(
            layers.Dropout(0.5)
        )

    model.add(layers.Dense(_output_num, name="wyjscie"))
    model.add(layers.Softmax())
    print(model.summary())

    # Model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model

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

print("y_test.shape: ", y_test.shape)
print("y_train.shape: ", y_train.shape)

# Display a sample photo
n = 800
plt.matshow(x_train[n])
plt.show()

# Create model
Model_CNN = create_cnn_model(
    _input_shape=x_test.shape[1:],
    _conv_num=[128, 256, 512, 256],
    _dense_num=[1024,512],
    _act="relu",
    _output_num=y_test.shape[1]
)

# Train
history = Model_CNN.fit(
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=200,
    validation_split=0.05
)

# Test
y_pred = Model_CNN.predict(x_test)
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Draw charts
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
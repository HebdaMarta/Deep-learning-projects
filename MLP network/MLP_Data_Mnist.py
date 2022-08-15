import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


def create_mlp_model(input_shape, dense_num, activation, output_num):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Flatten())
    # Dense layers
    for index, one_dense_units in enumerate(dense_num):
        model.add(
            layers.Dense(
                units=one_dense_units,
                activation=activation,
                name="dense_" + str(index)
            )
        )
        # Dropout
        model.add(layers.Dropout(0.4))

    model.add(layers.Dense(output_num, name="outgoing"))
    model.add(layers.Softmax())
    print(model.summary())

    # Model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

y_train = OneHotEncoder().fit_transform(y_train.reshape(-1,1)).toarray()
y_train = y_train.astype(np.float32)

y_test = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()
y_test = y_test.astype(np.float32)

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test.shape: ", y_test.shape)
print("y_train.shape: ", y_train.shape)

# Display a sample photo
n = 31000
plt.matshow(x_train[n])
plt.show()
print("one hot: ", y_train[n])
print("Label: ", np.argmax(y_train[n]))

# Create model
Model_MLP = create_mlp_model(
    input_shape=x_test.shape[1:],
    dense_num=[200,50],
    activation="relu",
    output_num=y_test.shape[1]
)

# Train
history = Model_MLP.fit(
    x=x_train,
    y=y_train,
    batch_size=200,
    epochs=20,
    validation_split=0.05
)

# Test
y_pred = Model_MLP.predict(x_test)
report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers
from keras.backend import l2_normalize
from keras.layers import Lambda
from keras.models import Model, load_model
from keras import regularizers
from sklearn.neighbors import NearestNeighbors
import os
import matplotlib.image as mpimg

# Load data
cifar = tf.keras.datasets.cifar10

(x_train, _), (x_test, _) = cifar.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

print("x_test.shape: ", x_test.shape)
print("x_train.shape: ", x_train.shape)

# Load model
old_model = load_model('/kaggle/input/cifarson')

out = old_model.get_layer('global_average_pooling2d').output
out = layers.Lambda(lambda x: l2_normalize(x, axis=1), name='encoder')(out)

encoder = Model(inputs=old_model.input, outputs=out, name='model')

encoder.summary()
encoder.compile(optimizer='rmsprop', loss='mse', metrics=["CosineSimilarity"])

# Testing the results

data_dir = '/kaggle/input/mojepliki/image_retrieval/'
codes = encoder.predict(x_train)
images = []

for img in os.listdir(data_dir):
    path = os.path.join(data_dir, img)
    query = mpimg.imread(path)
    images.append(query)

    query = query.reshape((1, 32, 32, 3))
    query = query / 255.0
    query = query.astype(np.float32)
    query_code = encoder.predict(query)

    nbrs = NearestNeighbors(n_neighbors=5, metric='manhattan', algorithm='kd_tree').fit(codes)

    distances, indices = nbrs.kneighbors(np.array(query_code))

    closest_images = x_train[indices]
    closest_images = closest_images.reshape(-1, 32, 32, 3)

    for i in range(5):
        closest_images[i] = closest_images[i].reshape((32, 32, 3))
        images.append(closest_images[i])

# View images

fig = plt.figure()

lista = [x for x in range(0, 61, 6)]
for indeks in range(10):
    a = lista[indeks]
    b = lista[indeks + 1]
    new_images = images[a:b]
    for i in range(6):
        ax = plt.subplot(1, 6, i + 1)
        plt.imshow(new_images[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    plt.show()
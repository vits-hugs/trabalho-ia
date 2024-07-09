import numpy as np                                                     # ndarrys for gridded data
import pandas as pd                                                    # DataFrames for tabular data
import os                                                              # set working directory, run executables
import matplotlib.pyplot as plt                                        # for plotting
import matplotlib.image as mpimg 
import seaborn as sns
import h5py   

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

import tensorflow as tf              # Importa TF2
from tensorflow import keras         # Importa Keras
from tensorflow.keras import layers  # Ferramentes do Keras mais usadas para acesso mais rápido
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical

filePath = "train_catvnoncat.h5"

with h5py.File(filePath, "r") as f:
    train_set_x = f["train_set_x"][()]
    train_set_y = f["train_set_y"][()]
    classes = list(f["list_classes"])

filePath = "test_catvnoncat.h5"

with h5py.File(filePath, "r") as f:
    test_set_x = f["test_set_x"][()]
    test_set_y = f["test_set_y"][()]

Xtrain = train_set_x
ytrain = train_set_y
Xtest = test_set_x
ytest = test_set_y

print(Xtrain.shape)
print(ytrain.shape)
print(Xtest.shape)
print(ytest.shape)
print(classes)

#normalização
Xtrain = Xtrain/255.0
Xtest = Xtest/255.0

#Exemplo do conjunto de treinamento
# print(Xtrain[1].shape)
# n=20
# plt.figure(figsize=(15, 15))
# for i in range(n):
#     ax = plt.subplot(4, 5, i + 1)
#     plt.imshow(Xtrain[i+10])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
# plt.show()

#one hot encoding da saida
num_classes = 2
ytrain = to_categorical(ytrain, num_classes)
ytest = to_categorical(ytest, num_classes)

#montando a rede neural
modelo = tf.keras.Sequential()
modelo.add(layers.Flatten())
modelo.add(layers.Dense(400, kernel_initializer="random_uniform", bias_initializer="random_uniform", activation="sigmoid"))
modelo.add(layers.Dense(2, kernel_initializer="random_uniform", bias_initializer="random_uniform", activation="softmax"))

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
modelo.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

input_shape = Xtrain.shape
modelo.build(input_shape)

modelo.summary()

#conjunto de validação
Xtr,Xval,ytr,yval = train_test_split(Xtrain,ytrain,test_size = 0.3)
num_train = np.size(Xtr,0)
print(num_train)

#treinamento
results = modelo.fit(Xtr, ytr, validation_data = (Xval, yval), batch_size = num_train, epochs=200, verbose=1)

ytestpred = modelo.predict(Xtest)
print('\nAccuracy: {:.4f}\n'.format(accuracy_score(ytest.argmax(axis=1), ytestpred.argmax(axis=1))))

#Your input to confusion_matrix must be an array of int not one hot encodings.
ConfusionMatrixDisplay.from_predictions(ytest.argmax(axis=1), ytestpred.argmax(axis=1))

print(modelo)
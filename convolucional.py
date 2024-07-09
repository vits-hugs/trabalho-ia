import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Carregar o conjunto de dados MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train  = h5py.File("train_catvnoncat.h5", 'r')
test  = h5py.File("test_catvnoncat.h5", 'r')
# (x_train, y_train), (x_test, y_test) = (train['train_set_x'], train['train_set_y']) , (test['test_set_x'],test['test_set_y'])
x_train = np.array(train['train_set_x']).astype('float32')/255
x_test = np.array(test['test_set_x']).astype('float32')/255

y_train = np.array(train['train_set_y'])
y_test = np.array(test['test_set_y'])

# Redimensionar as imagens para (28, 28, 1) e normalizar os valores dos pixels para o intervalo [0, 1]
# x_train = x_train.reshape((x_train.shape[0], 64, 64, 1)).astype('float32') / 255
# x_test = x_test.reshape((x_test.shape[0], 64, 64, 1)).astype('float32') / 255
print(y_test)
# Converter rótulos para one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

print(y_test)

# Construir a rede neural convolucional
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.1)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

import h5py
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#Fazer unpackinh

train_file = h5py.File("train_catvnoncat.h5", 'r')
test_file = h5py.File("test_catvnoncat.h5", 'r')
print(list(train_file.keys()))

images = train_file['train_set_x']
images = np.array([img.reshape(64 * 64 * 3) for img in images])
respostas = np.array(train_file['train_set_y'])
print("respostas", respostas)
perceptron = Perceptron(random_state=132)

perceptron.fit(images,respostas)

print(perceptron.score(images,respostas))

test_imgs = test_file['test_set_x']
test_imgs = np.array([img.reshape(64 * 64 * 3) for img in test_imgs])

test_resp = list(test_file['test_set_y'])

predic = perceptron.predict(test_imgs)

print(perceptron.score(test_imgs,test_resp))
ConfusionMatrixDisplay.from_predictions(test_resp, predic)
plt.show()
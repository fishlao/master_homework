"""
Use fast gradient sign method to craft adversarial on MNIST.

Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# 引入Tensorboard
from keras.callbacks import TensorBoard
# from tensorflow import keras
from fgsm import *


img_rows = 28
img_cols = 28
img_channel = 1
num_classes = 10
batch_size = 128
epochs = 1
eps = 0.5
print('\nLoading Fashion MNIST')

#preprocess data
fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

to_categorical = keras.utils.to_categorical
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, Y_train = X_train[ind], Y_train[ind]

# split the data to validation set 20% and trainning set 80%
VALIDATION_SPLIT = 0.2
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
Y_valid = Y_train[n:]
Y_train = Y_train[:n]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')

print('\nConstruction graph')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,
                 write_images=True)

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs, 
          verbose=1,
          validation_data=(X_valid, Y_valid),
          callbacks=[tbCallBack])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(X_test)
print("predictions 0", predictions[0])
print("labels 0", Y_test[0])
#loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label_array, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  d = img.reshape(28,28)
  #print("img 0", img)
  plt.imshow(d, cmap="gray")

  predicted_label = np.argmax(predictions_array)
  true_label = np.argmax(true_label_array)
  print("predicted_label true_label", predicted_label, true_label)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label_array = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  true_label = np.argmax(true_label_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, Y_test, X_test)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  Y_test)
# plt.show()

origin = X_test[0]
#x = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_channel), name='x')
xadv = tf.identity(origin)
print("xadv", xadv)
target = Y_test[0]
logits = predictions[0]
loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
dy_dx, = tf.gradients(loss, xadv)
xadv = tf.stop_gradient(xadv + eps*tf.sign(dy_dx))
print("xadv", xadv)
# xadv = tf.clip_by_value(xadv, 0.1, 1)

# plt.figure(figsize=(6,3))
# plt.subplot(2,2,1)
# plt.imshow(origin.reshape(28,28), cmap="gray")
# plt.subplot(2,2,2)
# plt.imshow(xadv.reshape(28,28), cmap="gray")
# plt.show()
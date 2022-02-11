from posixpath import split
from random import randrange
from urllib.parse import SplitResultBytes
import tensorflow as tf
import pandas as pd
import seaborn as sns
import PIL
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D
import keras.optimizers as optimizers
from sklearn.model_selection import train_test_split

def plot_acc(history, ax = None, xlabel = 'Epoch #'):
  # i'm sorry for this function's code. i am so sorry. 
  history = history.history
  history.update({'epoch':list(range(len(history['val_accuracy'])))})
  history = pd.DataFrame.from_dict(history)

  best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

  if not ax:
    f, ax = plt.subplots(1,1)
  sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
  sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
  ax.axhline(0.333, linestyle = '--',color='red', label = 'Chance')
  ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')  
  ax.legend(loc = 1)    
  ax.set_ylim([0.01, 1])

  ax.set_xlabel(xlabel)
  ax.set_ylabel('Accuracy (Fraction)')
  
  plt.show()

def show_img(label, index):
    img_ary = np.array(list(data_set.loc[data_set['label'] == label, 'image'])[index])
    img_ary = np.atleast_1d(img_ary)
    img_1d = img_ary[0].split()
    for i in range(0, len(img_1d)):
        img_1d[i] = int(img_1d[i])
    img_2d = np.reshape(img_1d, (48,48))
    print(img_2d)
    nparry = np.asarray(img_2d)
    print(img_2d)
    plt.imshow(nparry, cmap='gray', vmin=0, vmax=255)

def display_img():
    while True:
        rand_label = randrange(0,6,1)
        rand_index = randrange(0,100,1)
        show_img(rand_label, rand_index)
        plt.show()
        time.sleep(1)

def scalar_set_to_2dArray(set):
    result = []
    for row in set.itertuples():
        scalar = row.image
        scalar = scalar.split(" ")
        result.append(np.reshape(scalar, (48,48)))
    return np.asarray(result)

def set_to_1darray(set):
    result = []
    for row in set.itertuples():
        num = row.label
        result.append(num)
    return np.asarray(result)



dir_path = os.path.dirname(__file__)
file_path = os.path.join(dir_path, "train.csv")
data_set = pd.read_csv(file_path)
train, test = train_test_split(data_set, test_size = 0.2)


x_train = scalar_set_to_2dArray(train[['image']])
y_train = set_to_1darray(train[['label']])
x_test = scalar_set_to_2dArray(test[['image']])
y_test = set_to_1darray(test[['label']])

model = Sequential()
model.add(Conv2D(16, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dense(units = 128, activation = 'relu'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), shuffle=True)
plot_acc(history)
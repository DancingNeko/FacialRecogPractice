from random import randrange
import tensorflow as tf
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import numpy as np
import time

def show_img(label, index):
    img_ary = np.array(list(train_set.loc[train_set['label'] == label, 'image'])[index])
    img_ary = np.atleast_1d(img_ary)
    img_1d = img_ary[0].split()
    for i in range(0, len(img_1d)):
        img_1d[i] = int(img_1d[i])
    img_2d = np.reshape(img_1d, (48,48))
    print(img_2d)
    nparry = np.asarray(img_2d)
    print(img_2d)
    plt.imshow(nparry, cmap='gray', vmin=0, vmax=255)




file_path = "E:/Daniel/Machine Learning/Facial Emotion Rec/train.csv"
train_set = pd.read_csv(file_path)
train_set = train_set[['image','label']]

while True:
    rand_label = randrange(0,6,1)
    rand_index = randrange(0,100,1)
    show_img(rand_label, rand_index)
    plt.show()
    time.sleep(1)
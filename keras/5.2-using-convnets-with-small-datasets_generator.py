import os, shutil
import datetime

import os
from keras import layers, optimizers, losses, metrics, activations
from keras import  models
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator




train_dir = './datasets/cats_and_dogs_small/train/'
save_train_dir = './datasets/cats_and_dogs_small/gen/'
save_image_format = 'jpeg'
save_image_prefix = 'cats'

train_datagen = ImageDataGenerator(rotation_range=20)          # 이미지  vector value을 1/255 로 축소(0~1)

count = 0;
for name in  train_datagen.flow_from_directory(
                                directory=train_dir
                                # ,target_size=(150,150)
                                ,batch_size=10
                                ,save_to_dir=save_train_dir + save_image_prefix
                                ,save_format=save_image_format
                                ,save_prefix =save_image_prefix
                                ,classes=[save_image_prefix]):
    count += 1
    print('count:' , count)
    if count > 0 :
        break;



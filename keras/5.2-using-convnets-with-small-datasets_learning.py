import datetime

import os
from keras import layers, optimizers, losses, metrics, activations
from keras import  models
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation=activations.relu,input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation=activations.relu))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation=activations.relu))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation=activations.relu))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation=activations.relu))
model.add(layers.Dense(1,activation=activations.softmax))
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),loss=losses.binary_crossentropy,metrics=['acc'])
model.summary()

train_dir = './datasets/cats_and_dogs_small/train'
validation_dir = './datasets/cats_and_dogs_small/validation'

train_datagen = ImageDataGenerator(rescale=1./255)          # 이미지  vector value을 1/255 로 축소(0~1)
test_datagen = ImageDataGenerator(rescale=1./ 255)

train_generator = train_datagen.flow_from_directory(
    train_dir
    ,target_size=(150,150)
    ,batch_size=20
    ,class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
    validation_dir
    ,target_size=(150,150)
    ,batch_size=20
    ,class_mode='binary')

# for i , (data_batch, labels_batch) in enumerate(train_generator):
#     print('배치 데이터 크기:', data_batch.shape)
#     print('배치 레이블 크기:', labels_batch.shape)
#     break

model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)
model.save('./predict/5.2-using-convnets-with-small-datasets_learning/cats_and_dogs_small_1.h5')
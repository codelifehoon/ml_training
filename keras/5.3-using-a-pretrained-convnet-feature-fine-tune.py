import os
import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models, losses
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
# from matplotlib import  pyplot as plt


base_dir = './datasets/cats_and_dogs_small'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')




conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
conv_base.trainable = True


#fine tunning 을 위해서 기존 학습 model중 최상위 일부만 다시 학습할수 있도록 변경.
set_trainable = False
for layer in conv_base.layers:
    if layers.name == 'block5_conv1':
        set_trainable = True

    layer.trainable = set_trainable     # 차례대로 layer가 나온다면 block5_conv1~ 3까지가 마지막에 순차적으로 나올꺼요서..




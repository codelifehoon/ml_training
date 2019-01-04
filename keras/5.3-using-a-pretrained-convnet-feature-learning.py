import os
import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models, losses
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
# from matplotlib import  pyplot as plt




model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim = 4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss=losses.binary_crossentropy,metrics=['acc'])

history = model.fit(train_features,train_lables,epochs=30,batch_size=20,validation_data=(validation_features,validation_lables))


print('history keys:' , history.history)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))


plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Training val acc')
plt.title('accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# plt.close()







import tensorflow as tf
import  keras
import  numpy as np
import  pandas as pd

from  keras.datasets import mnist
from keras  import  models,layers
from keras.utils import  to_categorical
import datetime

start = datetime.datetime.now()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# pre-processing start
# modify training data for low cost.
x_train = x_train.reshape((60000,28*28))
x_train = x_train.astype(np.float32)/255        # 0~1 pixel value change 0~255 -> 0~1
x_test = x_test.reshape((10000,28*28))
x_test = x_test.astype(np.float32)/255

y_train_onehot = to_categorical(y_train)               # create one-hot encoding of the label
y_test_onehot = to_categorical(y_test)


# pre-processing end

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))


network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
network.fit(x_train,y_train_onehot,epochs=1,batch_size=128)

test_loss , test_acc =  network.evaluate(x_test,y_test_onehot)


x_predict = x_test[10:11]   # [sample,vectoes.] 구조가 되어야 함.  x_test[10] -> (784,) 안됨
y_predict = network.predict(x_predict)

end = datetime.datetime.now()

print(end - start, test_loss, test_acc)
print('y_predict:',y_predict)
print(np.argmax(y_predict,axis=1) , y_test[10])

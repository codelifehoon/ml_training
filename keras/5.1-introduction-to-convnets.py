import datetime
from keras import layers, optimizers, losses, metrics
from keras import  models
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf

tf.set_random_seed(777)
start_dt = datetime.datetime.now()

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3) , activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2,)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(),loss=losses.categorical_crossentropy,metrics=['accuracy'])
# model.summary()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))  # 1차원 자료를 60000 개의 28*28*1 의 이미지 자료로 분류
train_images = train_images.astype('float32') / 255      # iumage value를 0~1 사이로 변경

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)     # lable을 one-hot  변경
test_labels = to_categorical(test_labels)

model.fit(train_images,train_labels,epochs=5,batch_size=640)
end_dt = datetime.datetime.now()

print(end_dt-start_dt,model.evaluate(test_images,test_labels,verbose=0))

# 0:01:51.109513 [0.02573013983811288, 0.9923]
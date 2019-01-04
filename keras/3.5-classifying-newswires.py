import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers, losses
from keras.datasets import reuters
import pickle



print('tf.__version__:',tf.__version__)
tf.set_random_seed(777)
np.set_printoptions(threshold=np.nan)       # it will be large numpy not truncated. 1 2 3 ... 9998 9999
start_dt = datetime.datetime.now()
g_num_words = 10000


def vectorize_sequences(sequences, dimension=g_num_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def show_history(history):

    history_acc =  history.history['acc']
    history_loss = history.history['loss']
    history_val_acc = history.history['val_acc']
    history_val_loss= history.history['val_loss']
    epochs = range(1, len(history_acc) + 1)

    # f1 = plt.figure(figsize=(1, 1)) # plt-window size 설정
    plt.plot(epochs,history_loss,'bo',label='training loss')
    plt.plot(epochs,history_val_loss,'b',label ='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()   # 그래프를 초기화합니다
    plt.plot(epochs, history_acc, 'bo', label='Training acc')
    plt.plot(epochs, history_val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



# x_train:982   x_test:2246
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=g_num_words)

x_train_v = vectorize_sequences(x_train)
x_test_v = vectorize_sequences(x_test)

# # onehot with to_categorical : loss: 0.1108 - acc: 0.9598 - val_loss: 1.0660 - val_acc: 0.8050
y_train_onehot = y_train
y_test_onehot = y_test



x_val_v = x_train_v[:1000]
x_predict_v = x_train_v[1000:]
y_val_onehot = y_train_onehot[:1000]
y_predict_onehot = y_train_onehot[1000:]

# print(np.max(y_train))

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(),loss=losses.sparse_categorical_crossentropy,metrics=['accuracy'])
history  = model.fit(x_predict_v,y_predict_onehot,epochs=5,batch_size=10,validation_data=(x_val_v,y_val_onehot))

# show_history(history)

predictions = model.predict(x_test_v)
print(np.argmax(predictions[0:10],axis=1))
print(y_test[0:10])


print('*'*100)


with open("./predict/3.5-classifying-newswires-predict.dump", "wb") as fp:
    pickle.dump(model, fp)




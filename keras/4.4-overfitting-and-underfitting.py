import datetime
from keras import models, layers, optimizers, losses, metrics, regularizers
from keras.datasets import imdb
import numpy as np
import pandas as pd
import collections
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print(tf.__version__)
tf.set_random_seed(777)
g_num_words = 10000
start_dt = datetime.datetime.now()

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


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=g_num_words)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# g_num_words 크기의 matrix에 문자의 index 위치에 1 표시
x_train_v = vectorize_sequences(x_train)
x_test_v = vectorize_sequences(x_test)


y_train_v = np.asarray(y_train).astype(np.float32)   #  change value -> vector(as floatType)
y_test_v = np.asarray(y_test).astype(np.float32)


partial_index, val_index = train_test_split(np.array(range(x_train_v.shape[0])), shuffle=True,test_size=0.3, random_state=777)

x_val_v = x_train_v[val_index]
x_partial_train_v = x_train_v[partial_index]

y_val_v = y_train_v[val_index]
y_partial_train_v = y_train_v[partial_index]

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(g_num_words,),kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.1))  # 10%  dropout.
model.add(layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.1))  # 10%  dropout.
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=['accuracy'])

history = model.fit(x_partial_train_v,y_partial_train_v,epochs=4,batch_size=512,validation_data=(x_val_v,y_val_v))

# print(model.predict(x_test_v[:10]),y_test_v[:10])     # testdata predict.
# print(history.history.keys())


print(model.evaluate(x_test_v,y_test_v,verbose=0))
end_dt = datetime.datetime.now()
print(end_dt-start_dt)

# 1s 72us/step - loss: 0.2087 - acc: 0.9309 - val_loss: 0.2801 - val_acc: 0.8844
# 1s 72us/step - loss: 0.1677 - acc: 0.9444 - val_loss: 0.2832 - val_acc: 0.8837
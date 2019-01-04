import datetime
from keras import models, layers, optimizers, losses, metrics
from keras.datasets import imdb
import numpy as np
import pandas as pd
import collections
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


print(tf.__version__)
tf.set_random_seed(777)
g_num_words = 10000
start_dt = datetime.datetime.now()

def vectorize_sequences(sequences, dimension=g_num_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def vectorize_with_sklearn(sequences, dimension=g_num_words):
    np.set_printoptions(threshold=np.nan)       # it will be large numpy not truncated. 1 2 3 ... 9998 9999
    # one hot 형태의 matrix 구성을 위한 dimension 구성 ( 특정단어 위치에 count+ 위해서)
    val = np.arange(dimension)
    val = [np.array2string(val)]

    # learning data를 CountVectorizer 이용해 Vector 로 변하기 위해서 string으로 변환
    x_train_str = [np.array2string(np.asarray(x)) for x in sequences]
    cvector = CountVectorizer(lowercase=False,token_pattern='\\b\\w+\\b') # default token_pattern  removes tokens of a single char.
    cvector.fit(val)        #   create vocabulary dictionary of all token
    return cvector.transform(x_train_str).toarray()  # transform doc -> matrix

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

# g_num_words 크기의 matrix에 문자의 index 위치에 존재하는 수량만큼 +1 (변환이 많아서 느림)
# x_train_v = vectorize_with_sklearn(x_train)
# x_test_v = vectorize_with_sklearn(x_test)


y_train_v = np.asarray(y_train).astype(np.float32)   #  change value -> vector(as floatType)
y_test_v = np.asarray(y_test).astype(np.float32)


x_val_v = x_train_v[:10000]
x_partial_train_v = x_train_v[10000:]

y_val_v = y_train_v[:10000]
y_partial_train_v = y_train_v[10000:]


model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(g_num_words,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=['accuracy'])

history = model.fit(x_partial_train_v,y_partial_train_v,epochs=4,batch_size=512,validation_data=(x_val_v,y_val_v))
print(model.predict(x_test_v[:10]),y_test_v[:10])     # testdata predict.
# print(model.evaluate(x_test_v,y_test_v))
print(history.history.keys())

# display loss,acc metrix
# show_history(history)

end_dt = datetime.datetime.now()
print(end_dt-start_dt)

# original score : time: 0:00:15.482602 , loss: 0.1490 - binary_accuracy: 0.9555 - val_loss: 0.2807 - val_binary_accuracy: 0.8891

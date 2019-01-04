from keras import models, layers, optimizers, losses, metrics
from keras.datasets import imdb
import numpy as np
import pandas as pd
import collections
import tensorflow as tf
import datetime

tf.set_random_seed(777)
g_num_words = 10000
start_dt = datetime.datetime.now()

def vectorize_sequences(sequences, dimension=g_num_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=g_num_words)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])




x_train_v = vectorize_sequences(x_train)
x_test_v = vectorize_sequences(x_test)
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
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

model.fit(x_partial_train_v,y_partial_train_v,epochs=5,batch_size=512,validation_data=(x_val_v,y_val_v))

end_dt = datetime.datetime.now()


print(end_dt-start_dt)
# print(len(x_train))
# print(' '.join([reverse_word_index.get(i - 3,'?') for i in x_train[0]]))  # 문장의 단어 index가 word_index 값에 +3을 붙여서 되어서 변환시 빼줘야함.
# print(collections.OrderedDict(sorted(reverse_word_index.items())));

# for (key,value) in word_index.items():
#     print(key,value)

# print([max(sequence) for sequence in x_train])
# print(x_train[0],y_train[0])

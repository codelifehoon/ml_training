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


# x_train:982   x_test:2246
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=g_num_words)

x_test_v= vectorize_sequences(x_test)


model = None
with open("./predict/3.5-classifying-newswires-predict.dump", "rb") as fp:
    model = pickle.load(fp)



predictions = model.predict(x_test_v)
print(np.argmax(predictions[0:20],axis=1))
print(y_test[0:20])


print('*'*100)






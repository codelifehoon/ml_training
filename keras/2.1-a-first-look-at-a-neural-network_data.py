import tensorflow as tf
import  keras
import  numpy as np
import  pandas as pd
from  keras.datasets import mnist
from keras  import  models,layers
from keras.utils import  to_categorical
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()

digit = x_train[4]

fig, axes = plt.subplots(1, 4, figsize=(3, 3),
                         subplot_kw={'xticks': [], 'yticks': []})

axes[0].set_title("plt.cm.Blues")
axes[0].imshow(digit, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title("plt.cm.Blues_r")
axes[1].imshow(digit, interpolation='nearest', cmap=plt.cm.Blues_r)
axes[2].set_title("plt.BrBG")
axes[2].imshow(digit, interpolation='nearest', cmap='BrBG')
axes[3].set_title("plt.BrBG_r")
axes[3].imshow(digit, interpolation='nearest', cmap='BrBG_r')

plt.show()

# onehot encoding with keras.to_categorical
numbers = np.asarray([1, 2, 3, 4])
print(numbers)
print(to_categorical(numbers))   # <- the zero index is start from 0

# onehot encoding with sklearn OneHotEncoder
one_hot_number = numbers.reshape(4,1)
one = OneHotEncoder()
one.fit(one_hot_number)
print(one.transform(one_hot_number).toarray())  # <- a zero index starts at the first element








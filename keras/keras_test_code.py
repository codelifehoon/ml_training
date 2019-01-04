from keras.datasets import imdb
from keras import  preprocessing
import  numpy as np



data = np.array([[i] for i in range(500)])
data = np.reshape(data,[50,10])
data = preprocessing.sequence.pad_sequences(data,maxlen=15)
# print(data)


np2d = np.random.random((2,3))
np1d = np.random.random((3))

npmix = np2d + np1d
# print(np2d)
# print(np1d)
# print(npmix)



npzeros = np.zeros((2,5))
# print(npzeros)
# npzeros = npzeros.transpose()
# print(npzeros)


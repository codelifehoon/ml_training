from keras.datasets import imdb
from keras import  preprocessing
import  numpy as np

max_features= 1000
maxlen= 20

(x_train,y_train), (x_test,y_test) = imdb.load_data(num_words=max_features)  # 학습 자료를 읽을 때 가장 빈번한 상위 max_features의 데이터만 가져오도록. row에서 max_features에 포함안되는 단어는 제외되어서 전달됨


x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)





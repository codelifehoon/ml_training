from keras.datasets import imdb
from keras import preprocessing, models, optimizers, losses, metrics

# 특성으로 사용할 단어의 수
from keras.layers import Embedding, Dense, Flatten

max_features = 10000
# 사용할 텍스트의 길이(가장 빈번한 max_features 개의 단어만 사용합니다)
maxlen = 200

# 정수 리스트로 데이터를 로드합니다.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 리스트를 (samples, maxlen) 크기의 2D 정수 텐서로 변환합니다.
x_train_seq = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test_seq = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

print(x_train_seq)
print(y_train)

model = models.Sequential()
model.add(Embedding(10000,8,input_length=maxlen))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer=optimizers.RMSprop(),loss=losses.binary_crossentropy,metrics=['acc'])
model.fit(x_train_seq,y_train,epochs=10,batch_size=32,validation_split=0.2)

model_evaluate = model.evaluate(x_test_seq,y_test)
print(model_evaluate)


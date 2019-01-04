from keras.datasets import imdb
from keras import preprocessing, models, optimizers, losses, metrics, Sequential

# 특성으로 사용할 단어의 수
from keras.layers import Embedding, Dense, Flatten
import  os
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy as np
import  matplotlib.pyplot as plt

imdb_dir = './datasets/aclImdb'
train_dir = os.path.join(imdb_dir,'train')

labels = []
texts = []

for lable_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,lable_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            with open(os.path.join(dir_name,fname),encoding='utf8') as f :
                texts.append(f.read())

            if lable_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

print(len(labels), len(texts))


maxlen = 100  # 100개 단어 이후는 버립니다
training_samples = 200  # 훈련 샘플은 200개입니다
validation_samples = 10000  # 검증 샘플은 10,000개입니다
max_words = 10000  # 데이터셋에서 가장 빈도 높은 10,000개의 단어만 사용합니다

tokenizer = Tokenizer(max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print(texts[:1])
print('-'*100)
print(sequences[:1])
print('!!! finded {} unique token'.format(len(word_index)))

data = pad_sequences(sequences,maxlen=maxlen)

labels = np.asarray(labels)

print('shapes of data tensor:',data.shape)
print('shapes of lable tensor :', labels.shape)

indices = np.arange(data.shape[0])        # 전체 batch 만큼 숫자를 만들고 0~[0] 의 크기만늠
np.random.shuffle(indices)                # row indicator 를 random shuffle .
data = data[indices]
labels = labels[indices]


# 사전 훈련된 GloVe word embedding이 잘 되는지 확인 하기 위해서 train 은 200개만 가지고 해본다
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# GloVe 준비
glove_dir = './datasets/glove6B/'

embeddings_index = {}
with open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embeddings_index[word] = coefs

print('finded {} word vector.'.format(len(embeddings_index)))


## aclImdb에서 사용되는 10,000 개의 단어에 대해서 GloVe의 embedding 된 값을 부여한다(100차원)
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)       # glvoe의 embedding 값을 찾고
    if i < max_words:       # 0 ~ 9999
        if embedding_vector is not None:
            # 임베딩 인덱스에 없는 단어는 모두 0이 됩니다.
            embedding_matrix[i] = embedding_vector      # word_index 의  index번호로 glvoe-embedding값을 부여한다. 관련 문자의 값이 없ㅇ면 초기기값 0로 유지됨



## org model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

## layer[0] exchange with GloVe-model
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.summary()


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('./predict/6.1-using-word-embeddings-GloVe/pre_trained_glove_model.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
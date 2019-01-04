import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers, losses, metrics
from keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=np.nan)       # it will be large numpy not truncated. 1 2 3 ... 9998 9999
tf.set_random_seed(777)
start_dt = datetime.datetime.now()


def build_model(x_train):
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    # model.compile(optimizer=optimizers.RMSprop(),loss=losses.mse,metrics=[metrics.mae])
    model.compile(optimizer=optimizers.RMSprop(), loss='mse', metrics=['mae'])


    return model

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


k = 5
num_epochs = 50
all_scores = []
all_mae_histories = []
for i in range(k):
    print('처리중인 폴드 #', i)

    partial_index, val_index = train_test_split(np.array(range(x_train.shape[0])), shuffle=True,test_size=0.3, random_state=777)
    val_data    = x_train[val_index]
    val_targets = y_train[val_index]
    partial_x_train = x_train[partial_index]
    partial_y_train = y_train[partial_index]


    # 케라스 모델 구성(컴파일 포함)
    model = build_model(x_train)

    # 모델 훈련(verbose=0 이므로 훈련 과정이 출력되지 않습니다)
    model_history = model.fit(partial_x_train, partial_y_train,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1,verbose =0)

    mae_history = model_history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    # print('*'*100)
    # print(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(average_mae_history)
print('*'*100)
print(model.evaluate(x_test,y_test))
end_dt = datetime.datetime.now()
print(end_dt-start_dt)

# k-fold code 개발 [20.148067100375307, 2.766217970380596] 0:00:57.901924
# train_test_split 사용 [17.65613795261757, 2.7647230110916436] 0:00:54.321112



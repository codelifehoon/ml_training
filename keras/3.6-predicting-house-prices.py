import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers, losses, metrics
from keras.datasets import boston_housing
import  datetime


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



mean = x_train.mean(axis=0)     # 평균
x_train -= mean
std = x_train.std(axis=0)
x_train /= std                  # 표준편차

x_test -= mean
x_test /= std


k = 5
num_val_samples = len(x_train) // k
num_epochs = 50
all_scores = []
all_mae_histories = []
for i in range(k):
    print('처리중인 폴드 #', i)
    # 검증 데이터 준비: k번째 분할
    val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    # 훈련 데이터 준비: 다른 분할 전체
    partial_x_train = np.concatenate(
        [x_train[:i * num_val_samples],
         x_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_y_train = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
        axis=0)

    # print( i * num_val_samples,(i + 1) * num_val_samples)
    # print( 0, i * num_val_samples ,(i + 1) * num_val_samples,'max')

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

# plt.plot(range(0,len(average_mae_history)),average_mae_history)
# plt.xlabel('Epoches')
# plt.ylabel('Validation MAE')
# plt.show()


end_dt = datetime.datetime.now()
print(end_dt-start_dt)



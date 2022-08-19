# TensorBoard는 TensorFlow에 기록된 로그를 그래프로 시각화시켜서 보여주는 도구다.
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from _cffi_backend import callback

# 5명이 3번의 시험 점수로 다음 시험 점수 예측
x_data = np.array([[70,85,80],[71,89,78],[50,80,60],[66,20,60],[50,30,10]])
print(x_data)
y_data = np.array([73, 82, 72, 55, 34])

model = Sequential()
# model.add(Dense(1, input_dim = 3, activation='linear'))
model.add(Dense(6, input_dim = 3, activation='linear',name='a')) # 3개가 들어와서 6개가 나옴
model.add(Dense(3,  activation='linear',name='b'))
model.add(Dense(1,  activation='linear',name='c'))

print(model.summary())

opti = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer = opti, loss = 'mse', metrics=['mse']) # , run_eagerly=True 즉시 실행

from keras.callbacks import TensorBoard #callbacks 시스템에 의해 호출된다.
tb = TensorBoard(log_dir='.\\my', histogram_freq=1,
                         write_graph=True,
                         write_images=True)
 # 브라우저에서 실행됨(localhost)

history = model.fit(x_data, y_data, batch_size=1, epochs=50, verbose =1, 
                    callbacks=[tb])
# verbose =1 을 해야 그래프가 출력 됨

plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

loss_metrics = model.evaluate(x=x_data,y=y_data)
print('loss_metrics', loss_metrics)

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data)))
new_x_data = np.array([[44,55,10],[90,55,78]])
print('새로운 예측값: ',model.predict(new_x_data))
# Keras를 모듈을 사용해 DeepLearning 모델 네트워크 구성 샘플
# 논리회로 분류 

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from click.core import batch

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0]) # xor 게이트

model = Sequential()
model.add(Dense(units=3, input_dim=2, activation='relu')) # 백터곱 병렬 연산 - 완전 연결층
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=1, activation='sigmoid')) 
# 다항함수일땐 sigmoid # 중간연산자 는 relu 

print(model.summary())
# 파라미터 수 = (입력 자료수 + 1)* 출력수 ex) (2 + 1) * 3 , (3 + 1 ) * 3, (3 + 1) * 1

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x, y, epochs=10, batch_size=1,verbose=1) 
# epochs:학습반복횟수를 의미한다. batch_size: 몇개의 샘플로 가중치를 검증할껀지
print(history)
print(history.history['loss'])
print(history.history['accuracy'])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.show()

print(model.weights)
print()

pred = (model.predict(x) > 0.5).astype('int32')
print('예측결과: ', pred.flatten())
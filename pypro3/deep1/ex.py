import pandas as pd
import numpy as np

# Regression 
# 문제1)
#
# http://www.randomservices.org/random/data/Galton.txt
# data를 이용해 아버지 키로 아들의 키를 예측하는 회귀분석 모델을 작성하시오.
# train / test 분리
# Sequential api와 function api 를 사용해 모델을 만들어 보시오.
# train과 test의 mse를 시각화 하시오
#
# 새로운 아버지 키에 대한 자료로 아들의 키를 예측하시오.

df = pd.read_csv("Galtons Height Data.csv")
print(df.columns)
# Index(['Family', 'Father', 'Mother', 'Gender', 'Height', 'Kids'], dtype='object')
data= df[df['Gender'].str.contains('F')].index
df.drop(data, inplace=True)
# print(df.head(30))

df_x = df['Father']
df_y = df['Height']

# print(np.corrcoef(df_x,df_y)) # 0.39131736

from sklearn.model_selection._split import train_test_split
# # train / test 분리 = 오버피팅 방지 목적
x_train, x_test, y_train, y_test =train_test_split(df_x,df_y, test_size=0.3, random_state=0)

from keras.models import  Sequential
from keras.layers import Dense, Activation
from keras import optimizers

model = Sequential()
model.add(Dense(units = 5, input_dim = 1, activation='linear'))
model.add(Dense(units = 1, activation='linear'))

opti = optimizers.Adam(learning_rate = 0.1)
model.compile(optimizer=opti, loss='mse', metrics=['mse']) 

# 모델 학습
history = model.fit(x=x_train, y=y_train, batch_size=1, epochs=100, verbose=0)
loss_metrics = model.evaluate(x=x_train, y=y_train,batch_size=1, verbose=2)

print('loss_metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print('설명력 : ',r2_score(y_test, model.predict(x_test)))
print('실제값 : ', y_test.head(3))
print('예측값 : ', model.predict(x_test).flatten())

new_data = [76 , 70, 60]
print('새 점수 예측 결과 : ', model.predict(new_data).flatten())


# print('2) function API 사용 : 유연한 구조, 입력 데이터로부터 여러 층을 공유하거나 다양한 입출력 사용 가능 -------------')
# from keras.layers import Input
# from keras.models import Model
#
# # 각 층을 일종의 함수로써 처리를 함. 설계부분이 방법1과 다르다.
# inputs = Input(shape =(1,))
# # outputs = Dense(5, activation='linear')(inputs) # 이전 층 레이어를 다음 층 함수의 입력으로 사용
# output1 = Dense(5, activation='linear')(inputs)
# outputs = Dense(1, activation='linear')(output1)
# model2 = Model(inputs,outputs)
#
# # 이하는 방법1과 같음
#
# # 인풋 객체를 하나 만들고 Dense로 붙이는데 이전 네트워크값을 붙여줌 ()
# print(model2.summary())
#
# opti = optimizers.Adam(learning_rate = 0.01)
# model2.compile(optimizer=opti, loss='mse', metrics=['mse']) 
# # mse: 평균제곱오차, 추측값에 대한 정확성을 측정하는 방법
#
# history2 = model2.fit(x=x_train, y=y_train, batch_size=1, epochs=100, verbose=0)
#
# loss_metrics2 = model2.evaluate(x=x_train, y=y_train,batch_size=1, verbose=2)
# print('loss_metrics : ', loss_metrics2)
# print('설명력 : ',r2_score(y_test, model2.predict(x_test)))
# print('실제값 : ', y_test)
# print('예측값 : ', model2.predict(x_test).flatten())
#
# new_data = [76 , 70, 60]
# print('새 점수 예측 결과 : ', model.predict(new_data).flatten())


# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.plot(x_train, model.predict(x_train), 'b', x_train, y_train, 'ko') # train
plt.show()

plt.plot(x_test, model.predict(x_test), 'b', x_test, y_test, 'ko')  # test
plt.show()

plt.plot(history.history['mse'], label='평균제곱오차')
plt.xlabel('학습 횟수')
plt.show()


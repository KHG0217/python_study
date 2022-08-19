# 현재 자동차 가격 예측을 위한 다중선형회귀 분석

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import tensorflow as tf

train_df = pd.read_excel('https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx?raw=true',
                         sheet_name='train')

test_df = pd.read_excel('https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx?raw=true',
                         sheet_name='test')
print(train_df.head(1))
print(test_df.head(1))

x_train =train_df.drop(['가격'],axis=1) # 가격을 제외한 나머지 열을 feature
x_test = test_df.drop(['가격'],axis=1)
y_train =train_df[['가격']]
y_test =test_df[['가격']]

print(x_train.head(2))
print(y_test.head(2))

print(x_train.columns)
print(x_train.shape)
# print(x_train.describe())

print(set(x_train.종류)) # {'대형', '중형', '소형', '준중형'}
print(set(x_train.연료)) # {'디젤', 'LPG', '가솔린'}
print(set(x_train.변속기)) # {'수동', '자동'}

# 범주형 칼럼을 Dummy화 : LabelEncoder, OneHotEncoder
# make_column_transformer를 사용하여 특정 열에만 OneHotEncoder 적용(LabelEncoder 가능)
transformer=make_column_transformer((OneHotEncoder(), ['종류','연료','변속기']),
                                    remainder='passthrough')
# remainder='passthrough' or 'drop': passthrough[나머지 모든열들은 자동 전달] drop[나머지열 모두 제거]
transformer.fit(x_train)

# '종류','연료','변속기' 세 개의 칼럼이 참여해 병합되어 원핫 처리됨

x_train = transformer.transform(x_train) # 모든 칼럼이 표준화됨
x_test = transformer.transform(x_test)
print('make_column_transformer :', type(x_train), x_train[:2])
print(x_train.shape)
print(y_train.shape)

# model
input = tf.keras.layers.Input(shape=(16,))
net = tf.keras.layers.Dense(units=32, activation='relu')(input)
net = tf.keras.layers.Dense(units=32, activation='relu')(net)
net = tf.keras.layers.Dense(units=1)(net)
model = tf.keras.models.Model(input, net)

print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=2)
print('evaluate :',model.evaluate(x_test,y_test))

y_pred = model.predict(x_test)
print('예측값 : ', y_pred[:10].flatten())
print('실체값 : ', y_test[:5].values.flatten())

# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['mse'], 'r', label='mse')
plt.plot(history.history['val_mse'], 'b', label='val_mse')
plt.xlabel('epochs')
plt.legend()
plt.show()

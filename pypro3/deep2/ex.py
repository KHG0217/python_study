# 문제2) 21세 이상의 피마 인디언 여성의 당뇨병 발병 여부에 대한 dataset을 이용하여 
# 당뇨 판정을 위한 분류 모델을 작성한다.

import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import optimizers, Input

# data = pd.read_csv('../testdata/pima-indians-diabetes.data.csv', header=None)
# col_list = 'Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome'.split(',')
# data.columns = col_list
# print(data.head(2))
# print(data.info())
#
# x_data = data.drop('Outcome', axis=1)
# y_data = data['Outcome']
#
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
# print(x_train.shape, x_test.shape)      # (537, 8) (231, 8)
#
# # Sequential API
# model = Sequential()
# model.add(Dense(36, activation='relu', input_dim=x_train.shape[1]))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# print(model.summary())
#
# import os
# MODEL_DIR = './model2_sav/'
# if not os.path.exists(MODEL_DIR):
#     os.mkdir(MODEL_DIR)
#
# chkpoint = ModelCheckpoint(filepath=MODEL_DIR + "def.hdf5", monitor='val_loss', mode='auto',
#                             verbose=0, save_best_only=True)
#
# es = EarlyStopping(monitor='val_loss', mode='auto', patience=5)
#
# history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=1000, verbose=2,
#                     validation_split=0.2,
#                     callbacks=[chkpoint, es])
# loss, acc = model.evaluate(x_test, y_test, verbose=2)
# print('모델 분류 정확도: {:5.2f}%'.format(100*acc))
# print('모델 분류 loss: {}'.format(loss))
#
# # 새로운 값으로 예측
# from keras.models import load_model
# del model
# mymodel = load_model(MODEL_DIR + "def.hdf5")
# new_data = x_test[:3]
# pred = mymodel.predict(new_data)
# print('pred : ', np.where(pred > 0.5, 1, 0).flatten())
#
#
# print('------------')
# # Function API
# inputs = Input(shape=((x_data.shape[1],)))
# output1 = Dense(36, activation='relu')(inputs)
# output2 = Dense(8, activation='relu')(output1)
# output3 = Dense(1, activation='sigmoid')(output2)
# model2 = Model(inputs, output3)
#
# model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# print(model2.summary())
#
# history2 = model2.fit(x=x_train, y=y_train, batch_size=x_train.shape[1], shuffle=True, epochs=50, verbose=2)
# loss, acc = model2.evaluate(x_test, y_test, verbose=2)
# print('모델 분류 정확도: {:5.2f}%'.format(100*acc))
# print('모델 분류 loss: {}'.format(loss))
#
# # 시각화
# loss = history2.history['loss']
# acc = history2.history['acc']
#
# epoch_len = np.arange(len(loss))
#
# plt.plot(epoch_len, loss, c='b', label='loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc = 'best')
# plt.show()
#
# plt.plot(epoch_len, acc, c='r', label='acc', )
# plt.xlabel('epochs')
# plt.ylabel('acc')
# plt.legend(loc = 'best')
# plt.show()
#
#
# # 예측
# col_list.remove('Outcome')
# print(col_list)
# new_data = []
#
# # 키보드로 값 받기
# # ex) 6 148 72 35 0 33.6 0.62 50   실제값:1
# for v in col_list:
#     new_data.append(float(input('{} :'.format(v))))
# print('new_data : ', new_data)
#
# # 차원확대
# np_new_data = np.array(new_data).reshape(-1, len(new_data))
# print('new_data : ', np_new_data)
#
# pred = model2.predict(np_new_data)
# print(np.where(pred > 0.5, '당뇨 환자', '당뇨 환자 아님').flatten())

# 문제3) BMI 식으로 작성한 bmi.csv 파일을 이용하여 분류모델 작성 후 분류 작업을 진행한다.
#
# https://github.com/pykwon/python/blob/master/testdata_utf8/bmi.csv
#
# train/test 분리 작업을 수행.
#
# 평가 및 정확도 확인이 끝나면 모델을 저장하여, 저장된 모델로 새로운 데이터에 대한 분류작업을 실시한다.
#
# EarlyStopping, ModelCheckpoint 사용.
#
# 새로운 데이터, 즉 키와 몸무게는 키보드를 통해 입력하기로 한다. fat, normal, thin으로 분류


df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/bmi.csv")
print(df.head(3))

# x_data = df.drop('label', axis=1)
#
df['label'] = np.where(df['label'] =='thin',0,df['label'])
df['label'] = np.where(df['label'] =='normal',1,df['label'])
df['label'] = np.where(df['label'] =='fat',2,df['label'])
y_data = df['label']

y_data2 = y_data.values
# x_data2 = x_data.values
# print(y_data2)
# print(x_data2)

x_data =df[['height','weight']]
# df['label'] = np.where(df['label'] =='thin',0,df['label'])
# df['label'] = np.where(df['label'] =='normal',1,df['label'])
# df['label'] = np.where(df['label'] =='fat',2,df['label'])
# y_data = df['label'].astype('category').values 



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
print(x_train.shape, x_test.shape)      #(14000, 2) (6000, 2)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# model
model = Sequential()
model.add(Dense(32, input_shape=(8, ), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])

import os
MODEL_DIR = './model3_sav/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

chkpoint = ModelCheckpoint(filepath=MODEL_DIR + "def.hdf5", monitor='val_loss', mode='auto',
                            verbose=0, save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='auto', patience=5)



model.fit(x_train, y_train, batch_size=10, epochs=1000, verbose=2,
                    validation_split=0.3,
                    callbacks=[chkpoint, es])

# # 새로운 값으로 예측
from keras.models import load_model
del model

mymodel = load_model(MODEL_DIR + "def.hdf5")


# 예측
height=int(input("키 입력:"))
weight=int(input("몸무게 입력:"))

new_data = pd.DataFrame(
    {'height':[height], 'weight':[weight]})

# 차원확대
np_new_data = np.array(new_data).reshape(-1, len(new_data))
print('new_data : ', np_new_data)

pred = model.predict(np_new_data)
print("예측 label : ",pred)


# 문제4) testdata/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.
#
# * 변수 종류 *
#
# satisfaction_level : 직무 만족도
#
# last_eval‎uation : 마지막 평가점수
#
# number_project : 진행 프로젝트 수
#
# average_monthly_hours : 월평균 근무시간
#
# time_spend_company : 근속년수
#
# work_accident : 사건사고 여부(0: 없음, 1: 있음)
#
# left : 이직 여부(0: 잔류, 1: 이직)
#
# promotion_last_5years: 최근 5년간 승진여부(0: 승진 x, 1: 승진)
#
# sales : 부서
#
#
#
# salary : 임금 수준 (low, medium, high)
#
#
#
# 조건 : Randomforest 클래스로 중요 변수를 찾고, Keras 지원 딥러닝 모델을 사용하시오.
#
# Randomforest 모델과 Keras 지원 모델을 작성한 후 분류 정확도를 비교하시오.

import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../testdata/HR_comma_sep.csv')
print(data.head(2))
print(data.info())
# 이항 데이터 제거
data.drop(['left','promotion_last_5years','Work_accident'],axis = 1, inplace = True)

# x 전처리 sales category화, salary drop
pre1_x_data = data.drop('salary', axis=1)  # feature에서 제거
# pandas는 문자열을 object라는 자료형 사용. 문자열 값의 종류가 제한적일 때는 category를 사용할 수 있다. 메모리 절감 효과
sales_cate_data = pre1_x_data['sales'].astype('category')
print(sales_cate_data[:2])

x_data = pre1_x_data
x_data['sales'] = sales_cate_data.values.codes   # values.codes : category를 dummy화
print(x_data[:2])

# salary열은 label  - category 화 : LabelEncoder도 사용 가능
pre_y_data = data['salary'].astype('category').values
print(pre_y_data[:2])
print(pre_y_data.categories)    # Index(['high', 'low', 'medium'], dtype='object')
print(set(pre_y_data))  # {'high', 'low', 'medium'}
y_data = pre_y_data.codes
print(y_data[:5])    # [1 2 2 1 1 1 1 1 1 1]
print(set(y_data))   # {0, 1, 2}

# train/test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (10499, 6) (4500, 6) (10499,) (4500,)

# one-hot 
onehot_train_labels = tf.keras.utils.to_categorical(y_train)    # label에 대해 원핫 인코딩
onehot_test_labels = tf.keras.utils.to_categorical(y_test)
print(onehot_train_labels[:2])
print(onehot_test_labels[:2])

# Randomforest -----------------------
from sklearn.ensemble import RandomForestClassifier

rnd_model = RandomForestClassifier(n_estimators=500, criterion='entropy')
rnd_model.fit(x_train, onehot_train_labels)
pred = rnd_model.predict(x_test)
print('예측값 : ', [np.argmax(i) for i in pred[:3]])
print('실제값 : ', y_test[:3])

from sklearn.metrics import accuracy_score
print('RandomForestClassifier 정확도 : ', accuracy_score(onehot_test_labels, pred))  # 정확도:0.53

# 중요변수 확인
print('변수들 : ', x_data.columns)
print('특성(변수) 중요도 :\n{}'.format(rnd_model.feature_importances_))

def plot_feature_importances(model):  # 특성 중요도 시각화
    n_features = x_data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_data.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()

plot_feature_importances(rnd_model)  # average_monthly_hours, last_eval‎uation, satisfaction_level 이 중요변수

print("\n------ keras model ---------------------")
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x_data.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# 학습 도중 모델 저장
import os
model_dir = './tf10/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

modelpath = '{}salary.hdf5'.format(model_dir)

# print(modelpath)
chkpoint = ModelCheckpoint(filepath=modelpath, loss='loss', verbose=2, save_best_only=True)
es = EarlyStopping(monitor='loss', mode='auto', patience=10)

history = model.fit(x_train, onehot_train_labels, epochs=10000, batch_size = 256,
                    validation_split=0.2, verbose=2, shuffle=True, 
                    callbacks=[es, chkpoint])
print('eval:', model.evaluate(x_test, onehot_test_labels))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo--', label='train loss')
plt.plot(epochs, val_loss, 'r-', label='validation loss')
plt.xlabel('epochs')
plt.legend()
plt.show()



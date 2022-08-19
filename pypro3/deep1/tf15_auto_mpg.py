# 다중선형회귀분석 : 자동차 연비 예측

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers
pd.set_option('display.max_columns', None)
dataset=pd.read_csv("../testdata/auto-mpg.csv",na_values='?')
print(dataset.head(2))

# print(dataset.describe())

dataset = dataset.dropna(axis=0)

# print(dataset.describe())

del dataset['car name']
# print(dataset.head(2))

print(dataset.corr())

# 시각화
# sns.pairplot(dataset[['mpg','weight','horsepower','displacement']], diag_kind ='kde')
# plt.show()
print()

# train/test
train_dataset =dataset.sample(frac=0.7, random_state=123)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset.shape) # (274, 8)
print(test_dataset.shape) # (118, 8)

# 표준화 작업 : 수식을 사용 (요소값 - 평균) / 표준편차

train_stat = train_dataset.describe()
train_stat.pop('mpg') # mpg 열은 label로 사용함으로 제외
print(train_stat)

train_stat = train_stat.transpose()
print(train_stat)

# mpg
train_labels = train_dataset.pop('mpg')
print(train_labels[:2])

test_labels = test_dataset.pop('mpg')
print(test_labels[:2])

def std_func(x):
    return (x - train_stat['mean'] / train_stat['std'])

print(train_dataset[:3])
print(std_func(train_dataset[:3]))

# 표준화된 feature dataset
st_train_data = std_func(train_dataset)
st_test_data = std_func(test_dataset)

print('--model----')
from keras.models import Sequential
from keras.layers import Dense

def bulid_model():
    network = Sequential([
        Dense(units=64, activation='relu', input_shape=[7]), # 8개중 1개빠진 7
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    # opti = tf.keras.optimizers.RMSprop(0.001)
    opti = tf.keras.optimizers.Adam(0.001)
    network.compile(optimizer=opti, loss='mean_squared_error', metrics=['mean_absolute_error','mean_squared_error'])
    return network

model = bulid_model()
print(model.summary())

# fit 전에 모델로 predict 
print(model.predict(st_train_data[:1]))

#fit
epochs = 1000

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5)
history = model.fit(st_train_data,train_labels, batch_size=32, epochs=epochs, validation_split=0.2, verbose=2, callbacks=[es])

df = pd.DataFrame(history.history)
print(df.head(3))

def plot_history(history):

  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
  plt.legend()
  plt.show()
  
plot_history(history)

# evaluate
loss, mae, mae = model.evaluate(st_test_data, test_labels)
print('loss : {:5.5f}'.format(loss))
print('mae : {:5.5f}'.format(mae))
print('mae : {:5.5f}'.format(mae))

# predict
test_pred = model.predict(st_test_data)
print('예측값 :', test_pred)
print('실제값 :', test_labels)


# zoo dataset으로 동물의 type 분류

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from django.db.migrations import optimizer

xy = np.loadtxt('../testdata/zoo.csv', delimiter=',')
print(xy[0], xy.shape)  # (101, 17)

x_data = xy[:, 0:-1]    # feature
y_data = xy[:, -1]      # label
print(x_data[:3])
print(y_data[:3])
print(set(y_data))  # {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

nb_classes = 7
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print(y_one_hot[:3])

# model
model = Sequential()
model.add(Dense(32, input_shape=(16, ), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

# loss = 'sparse_categorical_crossentropy' : 내부적으로 원핫 처리를 함
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(x_data, y_data, epochs=100, batch_size=10, 
                    validation_split=0.3, verbose=0)
print('eval : ', model.evaluate(x_data, y_data))

# 시각화
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

import matplotlib.pyplot as plt
plt.plot(loss, 'b-', label='train_loss')
plt.plot(val_loss, 'r--', label='val_loss')
plt.legend()
plt.show()

plt.plot(acc, 'b-', label='train_loss')
plt.plot(val_acc, 'r--', label='acc_loss')
plt.legend()
plt.show()

# predict
pred_data = x_data[:1]
pred = np.argmax(model.predict(pred_data))
print('pred : ', pred)

print()
pred_datas = x_data[:5]
preds = [np.argmax(i) for i in model.predict(pred_datas)]
print('예측값들 : ', preds)
print('실제값들 : ', y_data[:5].flatten())


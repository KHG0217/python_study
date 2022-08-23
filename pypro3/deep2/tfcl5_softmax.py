import numpy as np
import matplotlib.pyplot as plt
 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
 
x = np.array([1.0,1.0,2.0])
 
y = softmax(x)
print(y) 
print(np.argmax(y))
print(np.sum(y))

print('다항분류 -----------')
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical # one - hot encoding을 지원
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
np.set_printoptions(suppress=True)

#dataset
xdata = np.random.random((1000, 12))
print(xdata[:2])

ydata = np.random.randint(5, size=(1000,1))
print(ydata[:2])
ydata =to_categorical(ydata, num_classes=5)
print(ydata[:3])
print([np.argmax(i) for i in ydata[:3]])

# model
model = Sequential()
model.add(Dense(units=32, input_shape=(12,), activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=5, activation='softmax')) # softmax 이면 loss= categorical_crossentropy
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(xdata, ydata, epochs=1000, batch_size = 32, verbose=0)
model_eval = model.evaluate(xdata,ydata, batch_size = 32, verbose=0)
print('평가 결과 :', model_eval)

# 시각화
plt.plot(history.history['loss'], label='loss')
plt.xlabel('epochs')
plt.legend(loc=1)
plt.show()

# 기존 값으로 예측
print('예측값 : ',model.predict(xdata[:5]))
print('예측값 : ',[np.argmax(i) for i in model.predict(xdata[:5])])

print('실제값 : ',ydata[:5])
print('실제값 : ',[np.argmax(i) for i in model.predict(ydata[:5])])

classes = np.array(['국어','영어','수학','과학','체육',])
print('예측값 : ',classes[np.argmax(model.predict(xdata[:5]), axis=-1)])
# 효율성 향상을 위한 네트워크 확장
# 개인이 연구에 의해 할 수 있으나 전문가들이 공개한 샘플 네트워크를 참조하면 효과적

import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') /255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') /255

# 시각화
plt.figure(figsize=(5, 5))
for c in range(16):
    plt.subplot(4, 4, c+1)
    plt.axis('off') # 축의눈금 안나오게 하기
    plt.imshow(x_train[c].reshape(28, 28), cmap='gray')

plt.show()

model = tf.keras.Sequential([
    layers.Conv2D(input_shape=(28,28,1), kernel_size=3, filters=16),
    layers.Conv2D(kernel_size=3, filters=32),
    layers.Conv2D(kernel_size=3, filters=64),
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=128, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2, 
                    validation_split=0.25)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='accuracy')
plt.legend()
plt.show()


print('eval : ',model.evaluate(x_test, y_test, verbose=0))

# 플립, 드랍아웃 추가
model = tf.keras.Sequential([
    layers.Conv2D(input_shape=(28,28,1), kernel_size=3, filters=16),
    layers.MaxPool2D(strides=(2,2)),
    layers.Conv2D(kernel_size=3, filters=32),
    layers.MaxPool2D(strides=(2,2)),
    layers.Conv2D(kernel_size=3, filters=64),
    layers.MaxPool2D(strides=(2,2)),
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(units=128, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2, 
                    validation_split=0.25)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='accuracy')
plt.legend()
plt.show()


print('eval : ',model.evaluate(x_test, y_test, verbose=0))

# 참고 소스 : VGGNet style 네트워크
# dataset은 Fashion MNIST with CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(28,28,1), kernel_size=(3,3), filters=32, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2, 
                    validation_split=0.25)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='accuracy')
plt.legend()
plt.show()


print('eval : ',model.evaluate(x_test, y_test, verbose=0))



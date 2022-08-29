# Fashion MNIST dataset

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

f_mnsi = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = f_mnsi.load_data()
print(train_images, train_labels.shape, test_images, test_labels.shape)

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
print(set(train_labels))
# {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# plt.imshow(train_images[0], cmap='gray')
# plt.show()

plt.figure(figsize=(10, 10))
for i in range (25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[train_labels[i]])
    plt.imshow(train_images[i], cmap='gray')
    
plt.show()

# 정규화
train_images = train_images / 255.0
test_images = test_images / 255.0

# model
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(units=128, activation=tf.nn.relu),
        keras.layers.Dense(units=64, activation=tf.nn.relu),
        keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=128, epochs=10, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('모델 성능 loss: ',test_loss)
print('모델 성능 test_acc: ',test_acc)

pred = model.predict(test_images)
print(pred[0])
print('예측값 : ', np.argmax(pred[0]))
print('실제값 : ', test_labels[0])

# 시각화
def plot_image_func(i, pred_arr, true_label, img):
    pred_arr, true_label, img = pred_arr[i], true_label[i], img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='Greys')
    
    pred_label = np.argmax(pred_arr)
    if pred_label == true_label:
        color='blue'
    else:
        color='red'
        
    plt.xlabel("{} {:2.0}% ({})".format(class_names[pred_label], 100*np.max(pred_arr),
                                        class_names[true_label]), color=color)
    
def plot_value_func(i, pred_arr, true_label):
    pred_arr, true_label = pred_arr[i], true_label[i]
    thisPlot = plt.bar(range(10), pred_arr)
    plt.ylim([0, 1])
    pred_label = np.argmax(pred_arr)
    thisPlot[pred_label].set_color('red') # 예측값
    thisPlot[pred_label].set_color('blue') # 실제값

i = 0
plt.figure(figsize=(6,3))

plt.subplot(1,2,1) # 1열은 이미지 출력
plot_image_func(i, pred, test_labels, test_images)

plt.subplot(1,2,2) # 2열은 막대그래프 출력
plot_value_func(i, pred, test_labels)
plt.show()
    
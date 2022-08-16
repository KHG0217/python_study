# tf.constant() : 텐서를 직접 기억
# tf.variable() : 텐서가 저장된 주소를 참조

import tensorflow as tf
import numpy as np

# Graph 영역 내에서 실행됨 - 속도가 빠름 
node1 = tf.constant(3,tf.float32)
node2 = tf.constant(4.0)
imsi = tf.add(node1, node2)
print(imsi)
print()
node3 = tf.Variable(3, dtype=tf.float32)
node4 = tf.Variable(4.0)
node4.assign_add(node3) # node4 = node4 _node3
print(node4)
print()

a = tf.constant(5)
b = tf.constant(6)
c = tf.multiply(a,b) #30
print(c, c.numpy())
result = tf.cond(a < b, lambda:tf.add(10,c), lambda:tf.square(a))
print(result)
print()

v= tf.Variable(1)

@tf.function
def find_next_func():
    v.assign(v + 1)
    if tf.equal(v % 2, 0):
        v.assign(v + 10)
        
find_next_func()
print(v.numpy())

print('-'*50)
def func1():    # 1 부터 3 까지 증가
    imsi = tf.constant(0)
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su) #tensor 영역내에선 가급적 이방법
        # imsi = imsi + su
        imsi += su
        
    return imsi
kbs = func1()
print(kbs.numpy(), ' ', np.array(kbs))

print('-'*50)
imsi = tf.constant(0)
@tf.function
def func2():    # 1 부터 3 까지 증가
    # imsi = tf.constant(0) # 안에있든 밖에있든 계산엔 상관 x
    global imsi
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su) #tensor 영역내에선 가급적 이방법
        # imsi = imsi + su
        imsi += su
        
    return imsi
mbc = func2()
print(mbc.numpy(), ' ', np.array(mbc))

print('-'*50)

imsi = tf.Variable(0) #@tf.function 일때 Variable 밖에 선언해야함
@tf.function
def func3():    # 1 부터 3 까지 증가
    # imsi = tf.Variable(0) # @tf.function 일때 Variable 안에 선언하면 X
    su = 1
    for _ in range(3):

        # imsi += su # 오류
        imsi.assign_add(su)
        
    return imsi

sbs = func3()
print(sbs.numpy(), ' ', np.array(sbs))
print()

print('------구구단-----')
@tf.function # 그래프 영역 함수
def gugu1(dan):
    su =tf.constant(0)
    for _ in range(9):
        su = tf.add(su,1)
        # print(su.numpy()) # 그래프 영역내에서 계산이기 때문에 tensor가 아니면 오류
        
        # print('{}*{}={:2}'.format(dan,su,dan * su)) 
        # 그래프 영역 함수 내에선 연산만가능 출력모양X(서식있는 출력X)
        
gugu1(3)
print()

# @tf.function
def gugu2(dan):
    for i in range(1, 10):
        result = tf.multiply(dan, i) # 원소곱 (요소간 곱) * tf.matmul() 행렬곱
        print('{}*{}={:2}'.format(dan,i,result)) 
        
gugu2(5)

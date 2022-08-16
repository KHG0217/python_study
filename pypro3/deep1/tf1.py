import tensorflow as tf
import re
# print(tf.__version__) # 2.9.1

print(tf.executing_eagerly()) 
# True : 즉시 시작이 가능하다 그래프 내에서 선언한 변수를 즉시 선언할 수 있게 되었다.

# 상수 정의(생성) : 상수 텐서를 생성
a = 3
print(type(a)) # <class 'int'>

a = tf.constant(3)
print(type(a)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
print(a)
print(tf.constant(1), tf.rank(tf.constant(1))) # 0-D tensor
print(tf.constant([1])) # 1-D tensor 1차원
print(tf.constant([[1]]), tf.constant([[1]]).get_shape()) # 2-D tensor 2차원
print()

a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = a + b
print(c, type(c))

c = tf.add(a,b)
print(c, type(c))

d = tf.constant([3])
print(c + d) # Broadcast 연산
print(tf.convert_to_tensor(7, dtype = tf.float32)) # tf.Tensor(7.0, shape=(), dtype=float32)
print(tf.cast(7, dtype=tf.float32))
print(tf.constant(7.0))
print(tf.constant(7, dtype=tf.float32))
print()

print('numpy의 ndarray의 tensor 사이에 type은 자동변환됨')
import numpy as np
arr = np.array([1,2])
print(arr, type(arr))
tfarr =tf.add(arr, 5) # tensor Type으로 형변환
print(tfarr)

print(tfarr.numpy()) # numpy type으로 형변환 (강제)
print(np.add(tfarr, 3)) # numpy type으로 형변환 (자동)
print()

print('---텐서형의 변수 선언---')
f = tf.Variable(1.0)
v = tf.Variable(tf.ones((2,)))
m = tf.Variable(tf.ones((2,1)))
print(f, v, m)
print(f.numpy()) # tensorflow 그래프 영역에서 빠져나오기
print()

v1 = tf.Variable(1) # tensorflow 영역
print(v1)
v1.assign(10)   # 변수에 새로운 값 할당
print(v1)
print()

v1=tf.Variable([3])
v2=tf.Variable([5])
v3 = v1 * v2 + 10
print(v3)
print()

print()
b = tf.Variable(5)

def func1(x):   # 일반함수
    return x + b  # -> 속도가 느리다 -> 이유 ? tensor Graph 영역에 나와서 
#                    -> 일반영역에서 수행하고 다시  tensor Graph 영역 들어감

print(func1(3))
print(type(func1)) # <class 'function'>

@tf.function # auto graph 기능 일반 파이썬 함수가 tensorflow의 그래프 영역내에서 호출 가능한 함수객체가 된다.
def func2(x):   # tensor Graph 영역에서 계산 -> 속도 향상
    return x + b 

print(type(func2)) # <class 'tensorflow.python.eager.def_function.Function'>
# eager = session이 돌고있다는 뜻
print(func2(3))
print()

print('균등분포 난수 발생')
rand = tf.random.uniform([1], 0, 1) # 균등분포를 따르는 난수
print(rand.numpy())
print()

print('정규분포 난수 발생')
rand2 = tf.random.normal([4], mean=0, stddev=1) # 정규분포를 따르는 난수
print(rand2.numpy())
print()

print('----------------텐서의 구조 --------------')
g1 = tf.Graph()

with g1.as_default():
    c1 = tf.constant(1, name='c_one')
    print(c1)
    print(type(c1))
    print(c1.op)
    print(g1.as_graph_def())

print('~'*100)    
g2 = tf.Graph()
with g2.as_default():
    v1 =tf.Variable(initial_value=1, name='v1')
    print(v1)
    print(type(v1))
    print(v1.op)
    print(g2.as_graph_def())
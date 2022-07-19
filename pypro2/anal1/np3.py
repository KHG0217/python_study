# 배열 연산 : 기본적인 수학함수는 배열에 요소별로 적용된다. 

import numpy as np

x = np.array([[1,2],[3,4]], dtype = np.float64) # int32 -> float64 형변환
y = np.arange(5,9).reshape((2,2))
y = y.astype(np.float32) # int32 -> float32 형변환
print(x, x.dtype)
print(y, y.dtype)

print()
print(x + y) # 요소별 합
print(np.add(x,y))


print()
print(x - y) # 요소별 차
print(np.subtract(x,y))

print()
print(x * y) # 요소별 곱, 하다마다 곱
print(np.multiply(x,y))

print()
print(x / y) # 요소별 나누기
print(np.divide(x,y))

print('행렬곱 : 내적 ')
v = np.array([9,10]) # 1차원 vector
w = np.array([11,12]) # 1차원 vector

print(v * w) # 일반곱셈(요소별 곱, 하다마다 곱) = 차원이 줄어들지 않는다 
print(v.dot(w)) # 1차원 vector에 대해 행렬 곱(내적) => scala (99 * 120)
print(np.dot(v,w)) # v[0] * w[0] + v[1] * w[1]

print()
print(x) # 2차원
print(v) # 1차원
print(x * v) # 요소별 곱, 결과는 큰 차원을 따름 = 2차원

print(x.dot(v))# 행렬곱(내적). 결과는 낮은 차원을 따름 = 1차원
# x = 1,2,3,4  v = 9,10  ㄱ 자 연산
# 행렬곱 = x[0,0] * v[0] + x[0,1] * v[1] = 29
#        x[1,0] * v[0] + x[1,1] * v[1] = 67
print(np.dot(x,v))

print()
print(x) # 2차원
print(y) # 2차원
print(x.dot(y)) # ㄱ연산 
# x[0,0] * y[0,0] + x[0,1] * y[1,0] = 19 
# x[0,0] * y[0,1] + x[0,1] * y[1,1] = 22
# x[1,0] * y[0,0] + x[1,1] * y[1,0] = 43
# x[1,0] * y[0,1] + x[1,1] * y[1,1] = 50

print(np.dot(x,y)) 

print()
print(x)
print(np.sum(x))
print(np.sum(x, axis = 0)) # 열방향 연산(axis = 0)
print(np.sum(x, axis = 1)) # 행방향 연산(axis = 1)

print(np.mean(x))
print(np.argmax(x)) # x값중에 가장 큰수
print(np.max(x)) # x값중에 가장 큰수
print(np.cumsum(x)) # 누적합 행 방향

print(x)
print(x.T) # 열을 행으로 바꿈 , 전치
print(x.transpose()) # 전치
print(x.swapaxes(0,1)) # 전치

# Broadcasting 연산 : 크기다 다른 배열 간의 연산 / 부족한 열과행을 채움
# 작은 배열과 큰 배열이 연산할 경우 작은 배열이 큰 배열의 크기만큼 연산에 반복적으로 참여한다.

x = np.arange(1,10).reshape(3,3)
y = np.array([1,2,3])
z = np.empty_like(x) # x행열과 크기가 같은 배열을 만들어 준다 / 안에 들어가있는 값은 쓰레기 값
print(x)
print(y)

# x와 y간 더하기 연산
for i in range(3):
    z[i] = x[i] + y
print(z)

kbs = x + y 
print(kbs)


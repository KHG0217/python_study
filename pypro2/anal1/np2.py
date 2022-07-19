# numpy 배열은 c 배열을 이용한 파이썬 객체

import numpy as np

ss = [1, 2.5, True, 'tom']
print(ss, type(ss))

# numpy의 배열로 변환 : 같은 type 자료로만 구성
ss2 =  np.array(ss)
print(ss2, type(ss2)) #ndarray 타입

# 메모리 비교 (list 와 ndarray 비교)
li = list(range(1,10))
print(li)
print(id(li), id(li[0]), id(li[1])) # 0,1,2 값에 각각 주소가 다르다 -> 각각 객체가 만들어지기떄문 ->다중 포인터 -> 그렇기 때문에 타입이 달라도 상관없다.
print(li * 10) # li요소를 10번 반복한다.

# li 요소와 각각  10을 곱한 결과를 얻고 싶다면, for
for i in li:
    print(i * 10, end = " " )
print()
print([i *10 for i in li])

print('-------')
num_arr = np.array(li) # array로 형 변환
print(id(num_arr), id(num_arr[0]), id(num_arr[1])) # 0,1,2의 주소가 같다. ->단일 포인터 -> 하나의 포인터에 들어가기 떄문에 타입이 같아야 한다.
print(num_arr * 10) # ,가 없다 = array / ,가 있다 = list

print('-----1차원 배열: vector------')
a = np.array([1,2,3]) # 상위 type int -> float -> complex -> str
print(a)
print(type(a), a.dtype, a.shape, a.ndim, a.size) # shape :차원의 요소의 갯수,  ndim: 차원수 , size:요소의 수
print(a[0],a[1],a[2])
a[0] = 5
print(a)

print('------2차원 배열 : matrinx----------')
b = np.array([[1,2,3],[4,5,6]])
print(b)
print(type(b), b.dtype, b.shape, b.ndim, b.size) # shape :차원의 요소의 갯수,  ndim: 차원수 , size:요소의 수
print(b[0,0],b[0,1],b[1,0])
print()
print(b[[0]].flatten()) # 다차원을 1차원으로 바꾸는 함수
print(b[[0]].ravel()) # 다차원을 1차원으로 바꾸는 함수

print()
c = np.zeros((2,3)) # 0으로 채워줌 2행 3열
print(c)
print()

d = np.ones((2,3)) # 1로 가득찬 2행 3열 
print(d)
print()

e = np.full((2,3), 7) # 7로 채워서 만들어 줌 2행 3열
print("e",e)
print()

f = np.eye(3) # 주대각을 1 로 만들어줘서 만들어 줌 2행 3열
print(f)
print()

print()
print(np.random.rand(5), np.mean(np.random.rand(50))) # 균등분포
print()
print(np.random.randn(5),np.mean(np.random.randn(50))) # 정규분포

print(np.random.normal(0, 1, (2,3))) # 2행 3열짜리
print()
np.random.seed(0)
x1 = np.random.randint(10,size=6) #1차원
x2 = np.random.randint(10,size=(3,4)) #2차원
x3 = np.random.randint(10,size=(3,4,5)) #3차원
print(x1, x1.ndim, x1.size)
print(x2, x2.ndim, x2.size)
print(x3, x3.ndim, x3.size)

print('------------')
a = np.array([1,2,3,4,5])
print(a[1])
print(a[1:5:2])
print(a[1:])
print(a[-2:])

b = a # 주소를 치환
b[0] = 77
print(a)
print(b)
del b

c = np.copy(a) # 복사본을 만드는 것.
c[0] = 88
print(a) # b에서 바꾼값이 a 에 들어감 -> 주소를 같이 쓰기 떄문
print(c)
del c
print(a)

print()
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a)
print(a[0]) # 벡터 [1 2 3 4]
print(a[0,0]) # 스칼라 1
print(a[[0]]) # 매트릭스 [[1 2 3 4]]
print(a[1:, 0:2]) # 1행 이후로 0열과 1열만 

# sub array
print()
print("a",a)

b = a[:2, 1:3] # 0,1행 and 2,3열만 출력
print("b",b)

print(b[0]) # b의 0행 -> 2,3
print(b[0,0]) # b의 0행 0열 2
b[0,0] = 99 # b값의 손을댄건 a값에도 손을 댄것.
print(b)
print(a)





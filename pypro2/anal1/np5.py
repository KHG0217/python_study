# 배열에서 조건 연산 where(조건, 참, 거짓)
import numpy as np

x = np.array([1,2,3])
y = np.array([4,5,6])
conditionData = np.array([True,False,True])
result = np.where(conditionData, x, y)
print(result)

print()
aa = np.where(x >= 2)
print(aa) # (array([1, 2], dtype=int64),) 인덱스 1번째와 2번째가 2보다 크거나 같다.
print(np.where(x >= 2, 'T','F'))
print(np.where(x >= 2, x, x + 100))

print()
bb = np.random.randn(4,4) # 정규분포를 따르는 난수 발생
print(bb)
print(np.where(bb > 0, 2,bb))

print() # 배열 결합
kbs = np.concatenate([x,y]) # concatenate 배열 결합
print(kbs)

print() # 배열 분활
x1,x2 = np.split(kbs, 2)
print(x1)
print(x2)

print()
a = np.arange(1,17).reshape(4,4) # 2 차원
print(a)

x1,x2 = np.hsplit(a,2) # 좌,우로 분리
print(x1)
print(x2)

x1,x2 = np.vsplit(a,2) # 상,하로 분리
print(x1)
print(x2)

# 표본추출 (sampling)
# 복원 / 비복원 추출

li = np.array([1,2,3,4,5,6,7])

# 복원 추출 : 항아리 안에 내용을 꺼내고 다시 집어넣는것 (모집단의 갯수가 계속 유지됨)/ 중복이 됨
for _ in range(5):
    print(li[np.random.randint(0, len(li) - 1)], end = ' ')
    
print()
import random    
# 비복원 추출 : 항아리 안에 내용을 꺼내고 빼놓고 다시 뽑는것 (모집단 갯수 유지 x) /중복 x
print(random.sample(list(li), k=5)) # list 타입으로 li를 5개 뽑음 
print()
print(random.sample(range(1,46),k=6)) # 1~45까지의 랜덤한 6개 숫자 중복x

# 복원 추출
print()
print(list(np.random.choice(range(1, 46), 6)))
print(list(np.random.choice(range(1, 46), 6, replace=True)))

# 비복원 추출
print(list(np.random.choice(range(1, 46), 6, replace=False)))

# 가중치를 부여한 random 추출
ar = 'air book cat d e f god'
ar = ar.split(' ')
print(ar)
print(np.random.choice(ar, 3, p=[0.1,0.1,0.1,0.1,0.1,0.1,0.4])) # 선택확률 0.4를준 god가 많이 나옴

# 문제 풀어보기

# 문제 1 
x1 = np.random.normal(0, 1, (5,4))
print(x1) # 5행 4열 구조 다차원
for i in range(4):
    print("{0}행 합계:{1}".format(i+1,np.sum(x1[i,]))) # 행 단위 합계
    print("{0}행 최댓값:{1}".format(i+1,np.max(x1[i,]))) # 행 단위 최댓값
print("문제1 선생님 답 ----------")
# 선생님 답  
data = np.random.randn(5, 4)
print(data)
print(data.sum(axis=1))
print(data.max(axis=1))
print(data.min(axis=1))

print()
i = 1

for r in data:
    print(str(i) + "행 합계 : ", r.sum())
    print(str(i) + "행 최댓값 : ", r.max())
    i += 1
    

print("------------")    
# 문제 2-1 X 정수 채워넣는 부분 for문으로 작성 !
x2 = np.zeros((6,6)) # 6행 6열 zero 행렬
print(x2)
x2 = np.arange(1,37).reshape(6,6) # 1~36 정수 채우기
print(x2)

print()
print(x2[1,]) # 2번째 행 전체 원소 출력하기
print()
print(x2[0:6,4]) # 5번째 열 전체 원소 출력하기
print()
print(x2[2:5,2:5]) # 15~29 까지 아래 처럼 출력하기

print("문제2 선생님 답 ----------")
# 문제 2-1 선생님 답

arr = np.zeros((6, 6))
cnt = 0
for i in range(6):
    for j in range(6):
        cnt +=1
        arr[i, j] = cnt

print(arr)
print()
print(arr[1, :]) # 2번째 행 전체 원소 출력하기
print(arr[:, 4]) # 5번째 열 전체 원소 출력하기
print(arr[2:5, 2:5]) #3,4,5행 3,4,5열 출력

print("------------")
#문제 2-2
x3 = np.zeros((6,4)) # 6행 4열 행렬
print(x3)

# 20 ~100 사이의 난수정수 6개 발생
ran = random.sample(range(20,101),k=6)
ran = list(ran) # list로 변환 
print(ran)
print()
# 행의 시작열에 난수 저장?
for r in range(len(x3)): #행의 갯수 6
    num = ran.pop(0) # 난수를 하나씩 뽑음.
    for e in range(len(x3[0])): # 열의 갯수 4
        x3[r,e] = num #ex r:0 일때 e:0,1,2,3 한번 돌때마다 num값 1씩 증가
        num += 1
print(x3)

# 4. 첫 번째 행에 1000,마지막 행에 6000으로 수정
x3[0,:] =1000 # x3[0][:]
x3[-1,:] = 6000
print(x3)

# 3) step3 : unifunc 관련문제
#   표준정규분포를 따르는 난수를 이용하여 4행 5열 구조의 다차원 배열을 생성한 후
#   아래와 같이 넘파이 내장함수(유니버설 함수)를 이용하여 기술통계량을 구하시오.
#   배열 요소의 누적합을 출력하시오. 
# <<출력 예시>>
# ~ 4행 5열 다차원 배열 ~
# [[ 0.56886895  2.27871787 -0.20665035 -1.67593523 -0.54286047]
#            ...
#  [ 0.05807754  0.63466469 -0.90317403  0.11848534  1.26334224]]

arr = np.random.rand(4,5)
print(arr)

#~ 출력 결과 ~
print("평균 :", arr.mean())
print("합계 :", arr.sum())
print("표준편차 :", arr.std())
print("분산 :", arr.var())
print("최댓값 :", arr.max())
print("최솟값 :", arr.min())
print("1사분위 수 :", np.percentile(arr, 25))
print("2사분위 수 :", np.percentile(arr, 50))
print("3사분위 수 :", np.percentile(arr, 75))
print("요소값 누적합 :", np.cumsum(arr))








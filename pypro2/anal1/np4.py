# 배열에 행 이나 열 추가
import numpy as np

aa = np.eye(3)  # 3행 3열 단위행렬 생성 (정방행렬)
print(aa)

bb = np.c_[aa, aa[2]] # 2열과 동일한 열을 추가
print(bb)

cc = np.r_[aa, [aa[2]]] # 2행과 동일한 행을 추가
print(cc)

print()

a = np.array([1,2,3]) # 1행 3열
print(a)
print(np.c_[a]) # 3행 1열
print(a.reshape(3,1)) # 3행 1열 (구조변경 명령어 reshape)

print('----------append, insert, delete ------------')
print(a)    #1 차원 [1 2 3]
b = np.append(a, [4,5])
print(b) # [1 2 3 4 5] 행방향
c = np.insert(a,0, [6,7]) # 0번째 자리에 (맨앞에) 6,7 이 들어감
print(c)
d = np.delete(a,1) # 1번째 자리 지움
print(d)

print()
aa = np.arange(1,10).reshape(3,3)
bb = np.arange(10,19).reshape(3,3)
print(aa) # 2차원
print(bb) # 2차원 

cc = np.append(aa,bb) # 1차원 / axis(작업방향)을 지정하지 않으면 2차원 배열이 1차원 배열이 됨 
print(cc) 

cc = np.append(aa,bb, axis=0) # 행방향 배열 쌓기
print("a",cc) 

cc = np.append(aa,bb, axis=1) # 열방향 배열 쌓기
print("b",cc) 

print()
print(np.delete(aa,1)) # aa 배열을 1차원으로 변환 후 1번 인덱스 값 삭제
print(np.delete(aa,1 , axis=0)) # aa의 1번 인덱스 행을 삭제
print(np.delete(aa,1 , axis=1)) # aa의 1번 인덱스 열을 삭제
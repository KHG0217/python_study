# 시각화 : matplotlib 라이브러리를 사용
# figure : 그래프(차트)가 그려지는 영역 
# 축 : axis /x축:xaxis / y추기yaxis / 눈금:tic / x,y값은 숫자만 들어갈 수 있음

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family = 'malgun gothic') # 그래프에 한글 깨질 때
plt.rcParams['axes.unicode_minus'] = False # 한글깨짐 방지후 음수깨짐 방지
'''
x = ['서울', '인천', '수원'] # list 0번째,1번째,2번째 = 숫자라고 생각 / tuple 가능 / set x
y = [5, 3, 7]

plt.xlim([-1, 3]) # 경계값을 바꿔줌 -1 ~ 3
plt.ylim([0, 10]) # 경계값을 바꿔줌 0~ 10
plt.yticks(list(range(0, 11, 3))) # 0~10 까지 3씩 증가하게 y라벨 표현
plt.plot(x,y)
plt.show()

'''
'''
data = np.arange(1, 11 , 2)
print(data)
plt.plot(data) # 값을 하나에만 주면 자동으로 y축에 들어감
               # x 축은 자동으로 구간이 들어감 
x = [0,1,2,3,4]
for a, b in zip(x,data):    #zip 쌍을 만들어주는 함수
    plt.text(a, b, str(b))
    
plt.show()
'''

'''
x = np.arange(10)
y = np.sin(x)
print(x,y)

# plt.plot(x ,y, 'bo') #bo = 파란 동그라미 그래프
# plt.plot(x ,y, 'r+')  #r+ = 빨간색 +
plt.plot(x ,y, 'r-.', linewidth = 3, markersize=12) 
plt.show()
'''

'''
# hold : 하나의 Figure내에 plot을 복수로 표현
x = np.arange(0, np.pi * 3, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.figure(figsize=(10,5))
plt.plot(x, y_sin, color='r') # 선그래프
plt.scatter(x, y_cos, color='b')# 산점도 (산포도)
plt.xlabel('x축')
plt.ylabel('사인&코사인')
plt.legend(['sine','cosine']) #범넬을 주기
plt.show()
'''
'''
# subplot : figure 영역을 여러 개로 분활
x = np.arange(10)
y = np.sin(x)

plt.subplot(2,1,1) # row ,cilumn, panel number
plt.plot(x)
plt.title('첫번째')


plt.subplot(2,1,2) # row ,cilumn, panel number
plt.plot(y)
plt.title('두번째')

plt.show()
'''

#

irum = ['a','b','c','d','e']
kor = [80, 50, 70, 70, 90]
eng = [60, 20, 80, 70, 50]
plt.plot(irum, kor, 'ro-')
plt.plot(irum, eng, 'bs--')
plt.ylim([0,100])
plt.title('시험점수')
plt.legend(['국어','영어'], loc=2) # loc 범넬의 위치 시계 반대방향 오른쪽위를 중심으로
plt.grid(True)


# 차트를 이미지로 저장하기
fig = plt.gcf()
plt.show()
fig.savefig('test.png')

# 이미지 파일 읽기
from matplotlib.pyplot import imread
img = imread('test.png')
plt.imshow(img)
plt.show()

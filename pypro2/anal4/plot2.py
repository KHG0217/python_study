# 차트 종류 경험하기

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family = 'malgun gothic') # 그래프에 한글 깨질 때
plt.rcParams['axes.unicode_minus'] = False # 한글깨짐 방지후 음수깨짐 방지

'''
x = np.arange(10)
y = np.sin(x)
z = np.cos(x)

# 차트 영역 객체 선언 방법 1
plt.figure()    # matplotlib 스타일의 인터페이스 
plt.subplot(2,1,1) # row ,cilumn, panel number
plt.plot(x, y)
plt.subplot(2,1,2)
plt.plot(x, z)
plt.show()

# 차트 영역 객체 선언 방법 1-1
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.hist(np.random.randn(10), bins = 5, alpha = 0.3) # alpha 투명도
ax2.plot(np.random.randn(10))
plt.show()

# 차트 영역 객체 선언 방법 2
fig, ax = plt.subplots(nrows=2, ncols=1) # 2행 1열짜리 / 객체지향 인터페이스
ax[0].plot(x, np.sin(x))
ax[1].plot(x ,np.cos(x))
plt.show()
'''


data = [50, 80, 100, 70, 90]
'''
# plt.bar(range(len(data)), data) # 세로막대
plt.barh(range(len(data)), data) # 가로막대
plt.show()


plt.pie(data,explode=(0,0.1,0,0,0.5), colors=['yellow', 'red', 'blue']) #pie 형태 /explode 튀어나온 정도
plt.show()


plt.boxplot(data) # 값이 너무 크거나,낮은 하나의 특정데이터 확인가능
plt.show()


plt.scatter(data,data) # 산점도
plt.show()


# DataFrame 자료로 그래프 그리기
import pandas as pd
fdata = pd.DataFrame(np.random.randn(1000, 4),
                     index=pd.date_range('1/1/2000',periods=1000),
                     columns=list('ABCD'))

fdata = fdata.cumsum()
plt.plot(fdata)
plt.show()

# pandas의 plot 기능
print(type(fdata)) # <class 'pandas.core.frame.DataFrame'>
fdata.plot(kind= 'box')
plt.xlabel('time')
plt.ylabel('data')
plt.show()
'''

# seaborn 모듈 : matplotlib의 기능을 추가한 시각화 package
import seaborn as sns

titanic = sns.load_dataset('titanic')
print(titanic.info())

# sns.displot(titanic['age'])
# sns.boxplot(y='age', data=titanic)
t_pivot = titanic.pivot_table(index='class',columns='sex',aggfunc='size') # 색이 진할수록 밀도가 높은것
sns.heatmap(t_pivot)
plt.show()

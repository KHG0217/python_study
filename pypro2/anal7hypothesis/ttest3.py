# 비(눈)이 올때의 매출/ 비(눈)이 오지 않을때의 매출의 평균 차이 검정

# 귀무 : 강수 여부에 따른 매출액에 차이가 없다.
# 대립 : 강수 여부에 따른 매출액에 차이가 있다.

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# 매출 데이터 읽기
sales_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tsales.csv",
                         dtype={'YMD':'object'}) # YMD 칼럼의 int type => object로 변환해 읽기
print(sales_data.head(3))
# print(sales_data.info())
print()

# 날씨 데이터 읽기
wt_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tweather.csv")
print(wt_data.head(3))
# print(wt_data.info())
print()

# wt_data의 날짜를 2018-06-01 ==> 20180601로 변환하기(병합 merge를 위해)
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-','')) # map = 함수를 실행해주는 함수
# print(wt_data.head(3))
print()

frame = sales_data.merge(wt_data, how='left',left_on='YMD', right_on='tm')
print(frame.head(3))
# print(frame.columns)
#'YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn','maxWs', 'avgWs', 'ddMes'

data = frame.iloc[:,[0,1,7,8]] # 모든행에 YMD, AMT, maxTa, sumRn
print(data.head(3))

# 결측치가 있는지 확인
# print(data.isnull().sum()) #결측치 없음

print('독립표본 t검정------------')
# print(data['sumRn'] > 0)

# data['rain_yn'] = (data['sumRn'] > 0).astype(int) # .astype(int) 비옴:1, 안옴:0
# print(data.head(3))

print(True *1,False *1) # 1 0
data['rain_yn'] = (data['sumRn'] > 0) *1 # 비옴:1, 안옴:0 같음
print(data.head(3))

# 매출액 비교 box plot으로 시각화
sp = np.array(data.iloc[:, [1,4]]) # AMT , rain_yn 만 추출
# print(sp) # 2차원 배열

tg1 = sp[sp[:, 1]==0,0] # 집단 1 : 비 안올 때 매출액 ,0:0번째 열 추출
tg2 = sp[sp[:, 1]==1,0] # 집단 2 : 비 올 때 매출액 ,0:0번째 열 추출
print(tg1[:3])
print(tg2[:3])

# plt.plot(tg1)
# plt.show()
#
# plt.plot(tg2)
# plt.show()

plt.boxplot([tg1, tg2])
plt.show()
print(np.mean(tg1), ' ',np.mean(tg2)) # 비가 안올때:761040.2542372881   비가 올떄:757331.5217391305
print()

# 정규성 확인
print(stats.shapiro(tg1).pvalue) # 0.056049469858407974 >0.05 만족
print(stats.shapiro(tg2).pvalue) # 0.882739782333374 > 0.05 만족
print()

# 등분산성
print(stats.levene(tg1, tg2).pvalue) # 0.7123452333011173 > 0.05 만족
print()

# 정규성 만족, 등분산성 만족

print(stats.ttest_ind(tg1,tg2, equal_var=True))
#Ttest_indResult(statistic=0.10109828602924716, pvalue=0.919534587722196)
# 해석 : pvalue >0.05 이므로 귀무가설 채택
#      # 귀무 : 강수 여부에 따른 매출액에 차이가 없다.


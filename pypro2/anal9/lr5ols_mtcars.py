# 선형회귀 : mtcars dataset 사용 ols() 사용
# ML 중 지도학습 : 귀납법적 추론방식을 사용 - 일반 사례를 수집해 법칙을 만듦

import statsmodels.api
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars)
print(mtcars.columns)
# print(mtcars.describe())
# print(mtcars.info())

# 선형회귀분석 시작 1. 상관관계 확인
print('상관관계 확인')
print(mtcars.corr())
print()

print(np.corrcoef(mtcars.hp,mtcars.mpg)) # 2개씩 보기 -0.77616837
print(np.corrcoef(mtcars.wt,mtcars.mpg)) # 2개씩 보기 -0.86765938

print('단순선형회귀---')
model1 = smf.ols('mpg ~ hp',data=mtcars).fit()
print(model1.summary())
#  Prob (F-statistic):    1.79e-07  1.확인 <0.05 작으니 유의함
# R-squared:      0.602 2.확인 독립변수가 종속변수를 60.2% 설명한다. >15%
#Intercept     30.0989 절편 (회귀계수)
#hp            -0.0682 (기울기)
#P>|t| hp 비교 <0.05 이면 자격 OK
print(model1.summary().tables[0]) # 위에것만 보기

# y = -0.0682  * x + 30.0989

print('다중선형회귀---') # 독립변수 2개이상
model2 = smf.ols('mpg ~ hp+wt',data=mtcars).fit()
print(model2.summary())
# Prob (F-statistic):           9.11e-12 <0.05 OK
# Adj. R-squared:                  0.815 81% >15% OK
# Intercept     37.2273    P>|t|0.000 <0.05 OK
# hp            -0.0318    P>|t|0.001 //    OK
# wt            -3.8778    P>|t|0.000 //    OK

# y = -0.0318  * x1(hp) + -3.8778 * x2(wt) + 37.2273

# 예측 :
new_pred = model2.predict({'hp':[110,150,90],'wt':[4,8,2]})
print(new_pred)



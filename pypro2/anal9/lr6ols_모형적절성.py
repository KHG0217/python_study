# *** 선형회귀분석의 기존 가정 충족 조건 ***
#
# . 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
#
# . 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
#
# . 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
#
# . 등분산성 : 그룹간의 분산이 유사해야 한다. 
#            독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
#
# . 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.

# 여러 매채에 광고비 사용에 따른 판매량 데이터
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

advdf = pd.read_csv("testdata/Advertising.csv", usecols=[1,2,3,4])
print(advdf.head(3), advdf.shape)
print(advdf.info())
print()

print('상관계수 확인')
print(advdf.loc[:,['sales','tv']].corr()) # 0.782224 양의 상관관계

lm = smf.ols(formula = 'sales ~ tv', data =advdf).fit()
print(lm.summary())
# Prob (F-statistic):           1.47e-42 <0.05 OK
# R-squared:                       0.612 >15% OK
# Intercept  7.0326     P>|t| 0.000 <0.05
# tv         0.0475     P>|t| 0.000 <0.05

"""
plt.scatter(advdf.tv,advdf.sales) # 독립변수 종속변수
plt.xlabel('tv')
plt.ylabel('sales')
# x = pd.DataFrame({'tv':[advdf.tv.min(), advdf.tv.max()]})

y_pred = lm.predict(advdf.tv) 
plt.plot(advdf.tv,y_pred, c='red')
plt.title('단순선형회귀')
plt.show()
"""

# 예측 : 새로운 tv값으로 sales를 예측
x_new = pd.DataFrame({'tv':[222.2, 55.5, 100.0]})
pred = lm.predict(x_new)
print('예측값 : ', pred.values)
# [17.59523505  9.67087709 11.78625759]

# 다중선형회귀
print(advdf.corr())

# newpaper는 sales와의 상관관계가 약함
lm_mul = smf.ols(formula ='sales ~tv + radio + newspaper',data = advdf).fit()
print(lm_mul.summary())
print()

# 모데에서 확인한 결과 newspaper: p -value > 0.05 이므로 newspaper는 독립변수에서 제외
lm_mul = smf.ols(formula ='sales ~tv + radio',data = advdf).fit()
print(lm_mul.summary())

# 예측 : 새로운 tv값으로 sales를 예측
x_new2 = pd.DataFrame({'tv':[222.2, 55.5, 100.0], 'radio':[30,40,50]})
pred2 = lm_mul.predict(x_new2)
print('예측값 : ', pred2.values)
# 예측값 :  [18.72764663 12.98026122 16.89629275]
print()

print('선형회귀분석의 기존 가정 충족 조건')

# 잔차 구하기
fitted = lm_mul.predict(advdf.iloc[:,0:2]) # newspaper 제외하고 예측값 얻기
# print(fitted)
residual = advdf['sales'] - fitted # 잔차
print(residual[:10].values) # 잔차 10개만 보기
print(np.mean(residual)) # 잔차의 평균은 0에 가까움

import seaborn as sns
print('선형성')
sns.regplot(fitted, residual, lowess =True, line_kws={'color':'red'}) # lowess - 로컬이 가능한 선형회귀로 만들어 줌
plt.plot([fitted.min(),fitted.max()], [0,0], '--', color='grey') # 구분선 긋기
plt.show() # sns.regplot 이 평평하지 않고 곡선 형태를 보이므로 선형성은 만족하지 않는다.

print('정규성 : Q-Q plot')
import scipy.stats

# 표본에 있는 z값을 계산함
sr = scipy.stats.zscore(residual) # () 에 잔차 넣어주기

(x, y),_ = scipy.stats.probplot(sr) # 가능성 차트

sns.scatterplot(x, y)
plt.plot([-3,3],[-3,3],'--', color='blue')
plt.show() # 정규성 : Q-Q plot 그래프 확인 가능

# 잔차의 정규성을 샤피로 검정으로 확인
print('샤피로 검정 : ', scipy.stats.shapiro(residual)) # 샤피로() 에 잔차 넣어주기
#  pvalue=4.190036317908152e-09 < 0.05 이므로 정규성 만족 x
# log를 취하는 등의 작업으로 정규분포를 따르도록 데이터 가공 필요

print('독립성 : 잔차가 자기상관(인접 관측치의 오차가 상관되어 있음)이 있는지 확인')
# Durbin-Watson 검정을 함. 잔차의 독립성 확인 0 ~ 4 값이 나오는데
#              2에 가까우면 자기상관이 없음 = 독립성을 만족
# Durbin-Watson:  2.081
print()

print('등분산성')
sns.regplot(fitted, np.sqrt(sr), lowess =True, line_kws={'color':'red'}) # lowess - 로컬이 가능한 선형회귀로 만들어 줌
plt.plot([fitted.min(),fitted.max()], [0,0], '--', color='grey') # 구분선 긋기
plt.show()
# 빨간색 실선이 수평하지 못하므로 등분산성을 만족하지 못함
# 이상값, 비선형 확인
# 정규성은 만족하나, 등분산성을 만족하지 않은 경우 가중회귀분석 추천
print()

print('다중공선성 : 독립변수 간 강한 상관관계 확인')
# 분산 팽창 인수 (VTF, Variance Infiation Factor)로 확인
# 10을 넘으면 다중 공선성이 발생하는 변수
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(variance_inflation_factor(advdf.values,1)) # 1 번째 TV, 12.570312383503682
print(variance_inflation_factor(advdf.values,2)) # 2 번째 radio, 3.1534983754953845
print()

# 극단치 확인
from statsmodels.stats.outliers_influence import OLSInfluence

cd,_= OLSInfluence(lm_mul).cooks_distance # 극단값을 나타내는 지표를 반환
print(cd.sort_values(ascending=False).head())

# 이상치 의심 데이터들
# 130    0.258065
# 5      0.123721
# 35     0.063065
# 178    0.061401
# 126    0.048958

import statsmodels.api as sm
sm.graphics.influence_plot(lm_mul, criterion='cooks')
plt.show()

print(advdf.iloc[[130, 5, 35]]) # 제외하기를 권장하는 행



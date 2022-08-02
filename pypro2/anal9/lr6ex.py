# 회귀분석 문제 3) 
#
# kaggle.com에서 Carseats.csv 파일을 다운 받아 Sales 변수에 영향을 
# 주는 변수들을 선택하여 선형회귀분석을 실시한다.
#
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
#
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
#
# 완성된 모델로 Sales를 예측.

import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
plt.rc('font', family='malgun gothic')

# Price 가 Sales 영향을 준다고 가정
data = pd.read_csv("testdata/Carseats.csv")
# 6 9 10 1
data = data.drop([data.columns[6],data.columns[9],data.columns[10]],axis=1)
print(data.columns)
print(data.head(3))
print('상관계수 확인')

print(data.loc[:,['Sales','Price']].corr()) # -0.444951 음의상관관계

lm = smf.ols(formula = 'Sales ~ Price', data =data).fit()

# 모델 저장 후 읽기
# import joblib
# joblib.dump(lm, 'aaa.model')
# del lm
# lm = joblib.load('aaa.model')

print(lm.summary())
# Prob (F-statistic):           7.62e-21 <0.05 OK
# R-squared:                       0.198 >15 % OK
# Intercept     13.6419     P>|t| <0.05 OK
# Price         -0.0531     P>|t| <0.05 OK

# 예측하기 ( 새로운 Price 값으로 Sales 예측하기)
x_new = pd.DataFrame({'Price':[140, 180, 200, 300 ,60]})
pred = lm.predict(x_new)
print('예측값 : ', pred.values)
# 예측값 :  [ 6.21169259  4.08877185  3.02731148 -2.27999037 10.45753407]

# 회귀분석모형의 적절성을 위한 조건도 체크

# 1.잔차 구하기
fitted = lm.predict(data.iloc[:,5]) # Price에 대한 Sales 예측값
residual = data['Sales'] - fitted
print(residual)
print(np.mean(residual))
#-5.451195050909519e-15 평균이 0에 가까움

# 2. 선형성

sns.regplot(fitted, residual, lowess =True, line_kws={'color':'red'}) # lowess - 로컬이 가능한 선형회귀로 만들어 줌
plt.plot([fitted.min(),fitted.max()], [0,0], '--', color='blue') # 구분선 긋기
plt.show() # sns.regplot 비교적 평평한 형태로 보이므로 선형성은 만족

# 3. 정규성
# 잔차의 정규성을 샤피로 검정으로 확인
print('샤피로 검정 : ', scipy.stats.shapiro(residual))
# pvalue=0.27001717686653137 >0.05 이므로 정규성 만족

# 4. 독립성
# Durbin-Watson:                   1.892 2에 가까우므로 만족

# 5. 등분산성
sr = scipy.stats.zscore(residual) # 표본에 있는 z 값 계산
sns.regplot(fitted, np.sqrt(sr), lowess =True, line_kws={'color':'red'})
plt.plot([fitted.min(),fitted.max()], [0,0], '--', color='blue')
plt.show()
# 빨간색 실선이 비교적 수평하지 하므로 등분산성을 만족


lm_mul = smf.ols(formula ='Sales ~Price + CompPrice',data = data).fit()

print(lm_mul.summary())

# Prob (F-statistic):           6.58e-39 <0.05 OK
# Adj. R-squared:                  0.355 >15%  OK

#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      6.2787      0.933      6.731      0.000       4.445       8.112
# Price         -0.0875      0.006    -14.788      0.000      -0.099      -0.076
# CompPrice      0.0908      0.009      9.941      0.000       0.073       0.109
# P>|t| OK 

# 6. 다중공선성
# 10을 넘으면 다중 공선성이 발생하는 변수


from statsmodels.stats.outliers_influence import OLSInfluence
cd,_= OLSInfluence(lm_mul).cooks_distance
print(cd.sort_values(ascending=False).head())

# 이상치 의심 데이터들
# 376    0.027820
# 367    0.023587
# 316    0.019616
# 70     0.017061
# 278    0.017057

import statsmodels.api as sm
sm.graphics.influence_plot(lm_mul, criterion='cooks')
plt.show()


print(data.iloc[[376, 367, 316, 70, 278]]) # 제외하기를 권장하는 행





# 파이썬이 지원하는 회귀분석 대표적인 방법 맛보기

# 회구분석: 독립변수(연속형)가 종속변수(연속형)에 얼마나 영향을 주는지 인과관계를 분석
# 정량적인 모델을 생성함

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np

# 방법 1 : make_refression 사용. model X

np.random.seed(12)
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print(x)
print(y)
print(coef) # 기울기: 89.47430739278907
# 수식(모델) : y = 89.47430739278907 *x + 100

# 모델에 의한 예측값
new_x = 1.234 # 한번도 경험하지 못한 새로운 x에 대하 y를 얻음
y_pred = 89.47430739278907 * new_x +100
print('예측값은 ',y_pred)

xx = x
yy = y

print()


print("방법2 : LinearRession 사용. model O")
from sklearn.linear_model import LinearRegression
model =LinearRegression()

# 선형회귀 모델이 완성
fit_model = model.fit(xx,yy) # 학습시키 (독립변수(x),종속변수(y))
print('기울기: ',fit_model.coef_)
print('절편: ', fit_model.intercept_)

# 예측값 지원 함수를 사용
# 2차원으로 학습했기때문에 예측값이 2차원이여야 한다. [[]] <- 2차원으로 만들기
y_pred = fit_model.predict([[new_x]]) 
print('예측값2은 ',y_pred)
print()

print("방법3 : ols 사용. model O")
import statsmodels.formula.api as smf
import  pandas as pd
print(type(xx),xx.shape) # ndarray, (5, 1) <- 2차원
x1 = xx.flatten() # 차원 축소
print(x1.shape)

y1 = yy
print(y1.shape)

data = np.array([x1,y1])
df = pd.DataFrame(data.T)
df.columns = ['x1','y1']
print(df.head(3))

model2 = smf.ols(formula='y1 ~ x1', data=df).fit() # R의 형식을 흉내냄
print(model2.summary()) # 모델 정보 확인

#Intercept(절편):    100.0000 /  x1의 기울기: 89.4743

new_df = pd.DataFrame({'x1':[-1.3, -0.5, 1.234]})
y_pred2 = model2.predict(new_df)
print('예측값3은 ',y_pred2) 
# 0    -16.316600
# 1     55.262846
# 2    210.411295
print()

print("방법4 : linregress 사용. model O")
from scipy import stats

model3 = stats.linregress(x1, y1)
print('기울기 : ',model3.slope)
print('절편 : ',model3.intercept)





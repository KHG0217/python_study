# 보스톤 집값 데이터로 선형/비선형 회귀모델 비교

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

df= pd.read_csv("testdata/housing.data", header=None, sep ='\s+') # sep ='\s+' 공백을 기준으로 자름
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head(3))
print(df.corr())

x = df[['LSTAT']].values
y = df['MEDV'].values
print(x.shape, y.shape)

model = LinearRegression()

# 다항 특성 만들기
quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)
print(x_quad[:2])
print(x_cubic[:2])

print('선형회귀')
# 선형회귀
model.fit(x,y)
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
y_lin_fit = model.predict(x_fit)
# print(y_lin_fit)
model_r2 =r2_score(y, model.predict(x))
print('model_r2 : ',model_r2) # 설명력 : 0.5441462975864799
print()

# 다항회귀식은 단항회귀식이 문제가 있을 것 같을때 사용
print('2차 다항식')
# 2차 다항식
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))
quad_r2 =r2_score(y, model.predict(x_quad))
print('quad_r2 : ',quad_r2) # 설명력 :  0.6407168971636611
print()

print('3차 다항식')
# 3차 다항식
model.fit(x_cubic, y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
cubic_r2 =r2_score(y, model.predict(x_cubic))
print('cubic_r2 : ',cubic_r2) # 설명력 :  0.6578476405895719

# 시각화
plt.scatter(x, y, label ='학습 데이터', c="lightgray")
#                                      $R^2=%.2f$'%model_r2 결정계수값 찍기
plt.scatter(x_fit, y_lin_fit, label ='선형회귀 데이터, $R^2=%.2f$'%model_r2, c="b", lw=2, linestyle = ':')
plt.scatter(x_fit, y_quad_fit, label ='다항회귀 데이터2,$R^2=%.2f$'%quad_r2, c="r", lw=2, linestyle = '-')
plt.scatter(x_fit, y_cubic_fit, label ='다항회귀 데이터3,$R^2=%.2f$'%cubic_r2, c="k", lw=2, linestyle = '--')
plt.xlabel('하위계층비율')
plt.ylabel('100달러 단위 주택가격')
plt.legend()
plt.show()

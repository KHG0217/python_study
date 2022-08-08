# 비선형 회귀모델 : 선형 가정이 어긋날 때 (정규성을 만족하지 못할 때)
# 대처할 수  있는 방법으로 다항회귀모델 가능
# 입력 데이터의 특징을 변환해서 선형모델 개선

import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])
# plt.scatter(x, y)
# plt.show()

# 선형회귀 모델 작성
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis] # np.newaxis 차원 확대 sklearn의 독립변수는 2차행렬을 요구
print(x, x.shape)
model = LinearRegression().fit(x, y)
ypred = model.predict(x)
print(ypred)
# plt.scatter(x, y)
# plt.scatter(x, ypred, c='red')
# plt.show()
print()

# 그래프를 그려보고 산포도 파악후 비선형회귀모델 작성할지 아닐지 선택해야함

# 비선형회귀 모델 작성(PolynomialFeatures)
from sklearn.preprocessing import PolynomialFeatures # 다항식 특징을 추가가능
poly = PolynomialFeatures(degree=2, include_bias = False) # degree 열 갯수 degree열의 갯수를 늘려서 할 수록 잔차가 줄어듬
# 잔차가 너무 줄어들면 오버피팅 되서 너무 높게 설정 x
x2 = poly.fit_transform(x) # 특징 행렬을 만들기
print(x2)

model2 = LinearRegression().fit(x2, y)
ypred2 = model2.predict(x2)
print(ypred2)

plt.scatter(x, y)
plt.plot(x, ypred2, c='red')
plt.show()

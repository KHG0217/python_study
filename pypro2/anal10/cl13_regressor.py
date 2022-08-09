# 분류 모델중 대다수는 Regression(정량적 예측)도 가능
# boston 집값 데이터 사용

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.keys())

# DataFrame에 담기
dfx = pd.DataFrame(boston.data, columns=boston.feature_names)
dfy = pd.DataFrame(boston.target, columns=['MEDV'])
df = pd.concat([dfx, dfy], axis=1)
print(df.head(3), df.shape) # (506, 14)

#상관계수 확인
print(df.corr())

# 종속변수로 사용할 MEDV열과 상관관계가 높은 RM, LSTAT 열로 시각화
cols=['MEDV','RM','LSTAT'] 
# sns.pairplot(df[cols])
# plt.show()

x = df[['LSTAT']].values
y = df['MEDV'].values
print(x[:3])
print(y[:3])
print()

# 실습 1 : DecisionTreeClassifier
model = DecisionTreeRegressor(criterion='squared_error').fit(x,y) # mse?
print('예측값 : ', model.predict(x)[:5])
print('실제값 : ', y[:5])
print('결점계수 : ', r2_score(y,model.predict(x))) # 실제값,예측값
print()

# 실습 2 : RandomForestClassifier
model2 = RandomForestRegressor(n_estimators=1000,criterion='squared_error').fit(x,y)
print('예측값 : ', model2.predict(x)[:5])
print('실제값 : ', y[:5])
print('결점계수 : ', r2_score(y,model2.predict(x))) # 실제값,예측값

# 단순회구분서기 ols()
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('testdata/drinking_water.csv')
print(df.head(3))

# 상관관계 확인
print(df.corr())

# 적절성, 만족도: 0.766
# 위 두 변수는 인과관계가 있다고 가정하고 회귀분석을 수행. x:적절성 , y:만족도

model = smf.ols(formula='만족도 ~ 적절성', data=df).fit()
print(model.summary())
# 수식 : y = 0.7393 * x + 0.7789

print(0.766853* 0.766853) 

# https://cafe.daum.net/flowlife/SBYs
# 0.588063523609 : 상관계수를 제곱한 값이 모델의 R-squared(결정계수, 설명력)이 됨
#Adj. R-squared:                  0.586 <- 수정된 R-squared는 독립변수가 두개이상
# 설명력 : 독립변수가 종속변수의 분산을 어느정도 설명하는지를 알려줌.
# 선형회귀모델의 성늘을 표현할때 사용함. 절대적으로 신뢰하진 x
# 15% 이상일 경우 모델을 신뢰하고 사용
# 이 모델은 58.8%의 설명력을 갖고있다.

# p벨류는 0.05보다 작으면 모델을 사용할 수 있다.

# p벨류는 =t값을 이용해서 구한다

# t값은= 표준오차로기울기(회귀계수)를 나눔 (표준오차/기울기) 로 구함

# 표준오차는 표본평균의 ???

# t값을 제곱한것이 f값

# f값은 모든 모델의 p값을 구해줌

print('회귀 계수(Intercept, slope) : ', model.params)
print('결정 계수(R squared) :',model.rsquared)
print('p-value :',model.pvalues)
print()

# 결과 예측
print(df.적절성[:5].values)
new_df = pd.DataFrame({'적절성':[4, 3, 4, 2, 6]})
pred = model.predict(new_df)
print('예측결과 : ', pred)
# 로지스틱 회귀분석
# 종속변수와 독립변수 간의 관계로 예측모델을 생성한다는 점에서 선형회귀분석과 유사하다 하지만
# 독립변수(x)에 의해 종속변수(y)의 범주로 분류한다는 측면에서 분류분석 방법이다.
# 분류 문제에서 선형예측애 시그모이드 함수를 적용하여 
# 가능한 각 불연속 라벨 값에 대한 확률을 생성하는모델로 이진분류 문제에 흔히 사용되지만
#  다중클래스 분류(다중 클래스 로지스틱 회귀 또는 다항회귀) 에도 사용될 수 있다.

# 독립변수 : 연송형, 종속변수 : 범주형

import math
from scipy.constants.constants import pt

def sigFunc(x):
    return 1 / (1+ math.exp(-x))

# 0~1 사이의 값으로 변환
print(sigFunc(3))
print(sigFunc(1))
print(sigFunc(-2))
print(sigFunc(-5))

# mtcars dataset 사용
import statsmodels.api as sm

mtcardata = sm.datasets.get_rdataset('mtcars')
print(mtcardata.keys())
mtcars= sm.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))

mtcar = mtcars.loc[:,['mpg','hp','am']]
print(mtcar.head(3))
print(mtcar['am'].unique()) # [1 0] <- 범주형 데이터로 쓰기 좋음

# 연비와 마력수(독립변수:연속형) 에 따른 변속기 분류(종속변수: 범주형) (수동,자동)
# 분류 모델 작성 1
import statsmodels.formula.api as smf
formula = 'am ~ hp + mpg'
model = smf.logit(formula = formula, data= mtcar).fit()
print(model) # BinaryResults 2진형태로 나오는 것을 알 수 있다.
print(model.summary()) # 여기선 P>|z| < 값이 0.05 인지만 확인

import numpy as np
# print('예측값 :',model.predict())
pred = model.predict(mtcar[:10])
print('예측값 :',pred)
print('예측값 :',np.around(pred.values)) # 0.5를 기준으로 0,1 출력
print('실제값 :',mtcar['am'][:10].values)
print()

conf_tab =model.pred_table()
print('confusion matrix : \n ', conf_tab)
 
 #  [[16.  3.] 맞은갯수 26 틀린갯수 9
 # [ 3. 10.]]
 
 # 모델의 분류 정확도 확인1
print('분류 정확도 :', (16 + 10) / len(mtcar))
print('분류 정확도 :', (conf_tab[0][0] + conf_tab[1][1])/ len(mtcar))

# 모델의 분류 정확도 확인 2
from sklearn.metrics import accuracy_score
pred2 = model.predict(mtcar)
print('분류 정확도 :',accuracy_score(mtcar['am'], np.around(pred2)))
print('-'*50)

# 분류 모델 작성 2 : glm()
model2 = smf.glm(formula = formula, data=mtcar, family=sm.families.Binomial()).fit()
print(model2)
print(model2.summary()) #  Generalized Linear Model Regression Results

pred2 = model2.predict(mtcar[:10])
print('예측값 :',np.around(pred2.values)) # 0.5를 기준으로 0,1 출력
print('실제값 :',mtcar['am'][:10].values)

pred2 = model2.predict(mtcar)
print('분류 정확도 :',accuracy_score(mtcar['am'], np.around(pred2)))
print()

print('새로운 값으로 분류 결과 보기')
newdf = mtcar.iloc[:2].copy()
print(newdf)
print()

newdf['mpg'] = [10, 30]
newdf['hp'] = [100, 120]
print(newdf)
new_pred = model2.predict(newdf)
print('new_pred: ',np.around(new_pred.values)) # new_pred:  [0. 1.]
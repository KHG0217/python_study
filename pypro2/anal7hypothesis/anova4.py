# Two-way ANOVA (이원배치 분산분석) : 목적이 되는 요인이 2개 이상인 경우
# 두 요인의 교호작용을 검정할 수 있는 특징이 있다.
import numpy as np
import scipy.stats as stats
import  pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import urllib.request
import statsmodels.api as sm

# 태아수와 관측자수에 따른 태아의 머리둘레 데이터 사용
# 귀무 : 태아수와 관측자수는 태아의 머리둘레 평균과 관련이 없다.
# 대립 : 태아수와 관측자수는 태아의 머리둘레 평균과 관련이 있다.

url = "https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3_2.txt"
data = pd.read_csv(urllib.request.urlopen(url))
print(data)

# 교호(상호) 작용은 뺴고 처리함
ols1 = ols("data['머리둘레'] ~ C(data['태아수']) + C(data['관측자수'])", data = data).fit()
result = sm.stats.anova_lm(ols1, typ = 2)
print(result)

# 교호(상호) 작용 처리함
ols1 = ols("머리둘레 ~ C(태아수) + C(관측자수) +C(태아수):C(관측자수)", data = data).fit()
result = sm.stats.anova_lm(ols1, typ = 2)
print(result)
print()
# 교호작용이 들어갔을때 그 p-value 값을 읽어 해석한다
#C(태아수):C(관측자수)    0.562222   6.0     1.222222  3.295509e-01

# 해석 : p-value : 3.295509e-01 > 0.05 이므로 귀무가설 채택
#     태아수와 관측자수는 태아의 머리둘레 평균과 관련이 없다.


# [ANOVA 예제 1]
#
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
#
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
#
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

# 귀무: 기름종류에 따라 흡수하는 기름의 평균은 차이가 없다.
# 대립: 기름종류에 따라 흡수하는 기름의 평균은 차이가 있다.

data ={
        'kind':[1,2,3,4,2,1,3,4,2,1,2,3,4,1,2,1,1,3,4,2],
        'quantity':[64,72,68,77,56,None,95,78,55,91,63,49,70,80,90,33,44,55,66,77]
    }

fram =pd.DataFrame(data)
# fram = fram.fillna(fram.quantity.mean())
fram=fram.where(pd.notnull(fram), fram.mean(), axis='columns')
print(fram)



lmodel = ols('quantity ~ C(kind)',fram).fit() # C반드시 붙여주기
print(anova_lm(lmodel))

# p-value : 0.848244 < 0.05 귀무가설 채택
# 귀무: 기름종류에 따라 흡수하는 기름의 평균은 차이가 없다.



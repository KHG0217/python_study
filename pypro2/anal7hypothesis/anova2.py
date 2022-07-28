# 일원분산분석
# 강남구에 있는 GS 편의점 3 개 지역 알바생의 급여에 대한 평균에 차이가 있는가 검정하기

# 귀무가설: GS 편의점 3 개 지역 알바생의 급여에 대한 평균에 차이가 없다. 
# 대립가설: GS 편의점 3 개 지역 알바생의 급여에 대한 평균에 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import urllib.request

url="https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3.txt"
# data = pd.read_csv(url, header=None)
# print(data.head(2), type(data)) # DataFrame
# print(data.describe())
# print()

data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',') # delimiter 구분자
print(data[:2],type(data)) # ndarray

# 세 개의 집단에 급여 자료의 평균
gr1 = data[data[:, 1] ==1, 0]
gr2 = data[data[:, 1] ==2, 0]
gr3 = data[data[:, 1] ==3, 0]
print(gr1, ' ', np.mean(gr1)) # 316.625
print(gr2, ' ', np.mean(gr2)) # 256.44444444444446
print(gr3, ' ', np.mean(gr3)) # 278.0
print()

print('정규성 확인') # 1개씩 검사        >0.05 모두만족
print(stats.shapiro(gr1).pvalue) # 0.3336853086948395
print(stats.shapiro(gr2).pvalue) # 0.6561065912246704
print(stats.shapiro(gr3).pvalue) # 0.832481324672699
print()

print('등분산성 확인')
print(stats.levene(gr1,gr2,gr3).pvalue) # 표본의 갯수가 적을때 사용 (지금은 적당하지 않음)
print(stats.bartlett(gr1,gr2,gr3).pvalue) # 0.3508032640105389 >0.05  등 분산성 만족

# 데이터의 산포도
plt.boxplot([gr1, gr2, gr3], showmeans=True) # showmeans=True 평균보기
plt.show()

# 일원분산분석 방법1
df = pd.DataFrame(data, columns=['pay','group'])
print(df.head(3))
lmodel = ols('pay ~ C(group)', data=df).fit() # C() <- 범주형이라는걸 알림 , #독립변수 ~ 종속변수 학습
print(anova_lm(lmodel, typ=1))
print()
# p-value : 0.043589 < 0.05 귀무가설 기각
# 대립가설: GS 편의점 3 개 지역 알바생의 급여에 대한 평균에 차이가 있다.

# 일원분산분석 방법2
f_statistic,p_value = stats.f_oneway(gr1,gr2,gr3) #세 개의 집단에 급여 자료의 평균
print('f_statistic,p_value',f_statistic,p_value)
# p-value : 0.043589334959178244 <0.05 귀무가설기각
# 대립가설: GS 편의점 3 개 지역 알바생의 급여에 대한 평균에 차이가 있다.

# 사후검정 - 각 그룹간의 평균의 차이을 확인하기 위해서 하는검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukeyResult = pairwise_tukeyhsd(endog=df.pay, groups = df.group)
print(tukeyResult)

tukeyResult.plot_simultaneous(xlabel = 'meas', ylabel = 'group')
plt.show()


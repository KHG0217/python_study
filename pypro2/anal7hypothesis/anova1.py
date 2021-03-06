# 세 개 이상의 모집단에 대한 가설검정 – 분산분석
# ‘분산분석’이라는 용어는 분산이 발생한 과정을 분석하여 
# 요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 
# 요인에 의한 분산이 의미 있는 크기를 크기를 가지는지를 검정하는 것을 의미한다.
# 세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 
# 제1종 오류가 증가하게 되어 문제가 발생한다.
# 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 이용하게 된다.
# * 서로 독립인 세 집단의 평균 차이 검정

# 분산분석의 전제조건
# - 독립성 : 각 집단은 서로 독립
# - 정규성 : 각 집단은 정규분포를 따른다.
# - 불편성 : 등분산성을 갖춰야 함. (편향된 데이터가 아니여야 한다.) 

# 집단 간 분산이 집단 내 분산보다 충분히 큰 것인가? 를 파악하는 것
# 일원분산분석 (one-way ANOVA) 복수 집단을 대상으로 집단을 구분하는 요인이 1개


# 실습) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시. 
# 세 집단의 평균차이 검정을 실시. three_sample.csv'

# 귀무 : 교육방법에 따른 실기시험 평균점수에 차이가 없다.
# 대립 : 교육방법에 따른 실기시험 평균점수에 차이가 있다.

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

data = pd.read_csv("testdata/three_sample.csv")
print(data.head(3))
print(data.shape) # (80, 4)
print(data.describe())
print()

# plt.boxplot(data.score)
# plt.show() # score에 이상치가 있음을 확인

data = data.query('score <= 100')   # score 값 100이하만 취함
print(len(data)) # 78개 이상치값 2개가 빠졌음을 알 수 있다.



result = data[['method','score']]
m1 = result[result['method']==1]
m2 = result[result['method']==2]
m3 = result[result['method']==3]
score1 = m1['score']
score2 = m2['score']
score3 = m3['score']
print(score1[:2]) # 교육방법 1 score
print(score2[:2]) # 교육방법 2 score
print(score3[:2]) # 교육방법 3 score
print()


# 정규성 검정
print('정규성 검정')
print(stats.ks_2samp(score1,score2).pvalue) # 0.3096879629846001 > 0.05 만족
print(stats.ks_2samp(score1,score3).pvalue) # 0.7162094473752455 > 0.05 만족
print(stats.ks_2samp(score2,score3).pvalue) # 0.7724081666033108 > 0.05 만족
print()

# 정규성을 만족하면 anova, 만족하지 않으면 kruskal-wallis test

print('등분산성 검정')
print(stats.levene(score1, score2, score3).pvalue) # 0.11322850654055751 > 0.05 만족

# 정규성을 만족하면 anova, 만족하지 않으면 welch_anova test

# ANOVA 검정
import statsmodels.api as sm
print('평균1 : ', score1.mean()) # 67.38
print('평균2 : ', score2.mean()) # 68.35
print('평균3 : ', score3.mean()) # 68.87

lm = ols("data['score'] ~ data['method']", data = data).fit()
result = sm.stats.anova_lm(lm, typ=2)
print(result) # p -value : 0.727597

# 해석 : p -value : 0.727597 > 0.05 이므로 귀무가설 채택
# 귀무 : 교육방법에 따른 실기시험 평균점수에 차이가 없다.

print('사후검정 ---')
# ANOVA는 그룹간의 평균값 차이의 유의함을 판단해 준다.
# 그런데 각 그룹의 평균값의 차이는 설명하지 않음.
# 그래서 사후검정을 실시한다.(Post Hoc Test)을 실시함

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tr = pairwise_tukeyhsd(endog = data.score, groups = data.method)
print(tr) # reject: 유의미한 차이가 있다 ? True, / 없다 ? False
# 유의미한차이가 없다.

# 시각화 
tr.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()

# 가설검정 정리 : chi2, t-test, ANOVA
# jikwon 테이블을 사용
import MySQLdb
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('connection err: ', e)
    
conn = MySQLdb.connect(**config)
cursor = conn.cursor()

print('교차분석(이원카이제곱검정): 각 부서(범주형)와 직원평가점수(범주형)간의 관련성 여부')
# 귀무가설 : 각 부서와 직원평가점수 간에 관련이 없다.
# 대립가설 : 각 부서와 직원평가점수 간에 관련이 있다.

df = pd.read_sql("select * from jikwon", conn)
print(df.head(2))
buser = df['buser_num']
rating = df['jikwon_rating']

# 이원카이제곱 -> 교차표 작성
ctab = pd.crosstab(buser,rating)
print(ctab)

chi, p, df, _ =stats.chi2_contingency(ctab)
print('chi:{}, p:{}, df:{}'.format(chi, p, df))
# chi:7.339285714285714(카이제곱), p:0.2906064076671985(유의확률), df:6(자유도)
# 해석하는법 : 카이제곱과 임계값을 구하여 비교 
#                    카이제곱 < 임계값 귀무가설 채택
#                    카이제곱 > 임계값 대립가설 채택

#            p:0.2906(유의확률) >0.05(유의수준) 이므로 귀무가설 채택, 대립가설 기각
# # 귀무가설 : 각 부서와 직원평가점수 간에 관련이 없다.
print()

print('평균차이분석(t-test): 10,20번 부서와(범주형) 평균연봉(연속형) 값의 차이가 있는가') # 독립변수 범주형과 종속변수 연속형일때
# 귀무가설 : 두 부서간 연봉 평균의 차이가 없다.
# 대립가설 : 두 부서간 연봉 평균의 차이가 있다.
df_10 = pd.read_sql("select buser_num, jikwon_pay from jikwon where buser_num=10", conn)
df_20 = pd.read_sql("select buser_num, jikwon_pay from jikwon where buser_num=20", conn)
buser10 = df_10['jikwon_pay']
buser20 = df_20['jikwon_pay']
print()

print('평균: ',np.mean(buser10), ' ',np.mean(buser20))
# 부서 10 : 5414.285714285715
# 부서 20 : 4908.333333333333

# 정규성,등분산성 검정 (여기선 생략)

t_result = stats.ttest_ind(buser10,buser20)
print(t_result)
# Ttest_indResult(statistic=0.4585177708256519, pvalue=0.6523879191675446)
# 해석: pvalue=0.65238 >0.05 이므로 귀무가설 채택
#      귀무가설 : 두 부서간 연봉 평균의 차이가 없다.
print()

print('분산분석(ANOVA): 각 부서(범주형. 요인 1개 (그룹4개) )와 평균연봉(연속형) 값의 차이가 있는가') # 독립변수 범주형과 종속변수 연속형일때
# 귀무가설 : 4개의 부서간 연봉 평균차이가 없다.
# 대립가설 : 4개의 부서간 연봉 평균차이가 있다.
df3 = pd.read_sql("select buser_num, jikwon_pay from jikwon", conn)
buser = df3['buser_num']
pay = df3['jikwon_pay']

# 강 부서간 연봉 평균 차이 있는지 그래프로 확인해보기
gr1 = df3[df3['buser_num'] == 10]['jikwon_pay']
gr2 = df3[df3['buser_num'] == 20]['jikwon_pay']
gr3 = df3[df3['buser_num'] == 30]['jikwon_pay']
gr4 = df3[df3['buser_num'] == 40]['jikwon_pay']

plt.boxplot([gr1,gr2,gr3,gr4])
plt.show()

# 방법 1 
f_sta,pv = stats.f_oneway(gr1,gr2,gr3,gr4)
print(f_sta,pv)
# p- value:0.7454421884076983 >0.05 이므로 귀무가설 채택 대립가설 기각
# 귀무가설 : 4개의 부서간 연봉 평균차이가 없다.

# 방법2
lm = ols('jikwon_pay ~ C(buser_num)', data = df3).fit() # fit <- 학습
result = anova_lm(lm, typ=2)
print(result)
# p- value:0.745442 >0.05 이므로 귀무가설 채택 대립가설 기각
# 귀무가설 : 4개의 부서간 연봉 평균차이가 없다.
print()

print('사후 검정')
from statsmodels.stats.multicomp import pairwise_tukeyhsd
pt = pairwise_tukeyhsd(df3.jikwon_pay,df3.buser_num)
print(pt)
# reject 모두 False 임을 알 수있다 = 차이가 없다.

pt.plot_simultaneous()
plt.show()


# 집단 간 차이분석: 평균 또는 비율 차이를 분석
# : 모집단에서 추출한 표본정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있다.
# * T-test와 ANOVA의 차이
# - 두 집단 이하의 변수에 대한 평균차이를 검정할 경우 T-test를 사용하여 검정통계량 T값을 구해 가설검정을 한다.
# - 세 집단 이상의 변수에 대한 평균차이를 검정할 경우에는 ANOVA를 이용하여 검정통계량 F값을 구해 가설검정을 한다.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 단일 표본 t검정 : one-smaple t-test. 정규분포의 표본에 대해 기댓값을 조사하는 검정방법
# 실습1 : 어느 남성 집단의 평균키 검정 
# 귀무가설 : 어느 남성 집단의 평큔키는 175.0다.
# 상대가설 : 어느 남성 집단의 평균키는 175.0가 아니다.

one_sample = [167.0, 182.7, 169.6, 176.8, 185.0]
print(np.array(one_sample).mean()) # 176.21999999999997

result = stats.ttest_1samp(one_sample, popmean=175.0) # 이미 알려져있는 평균값:popmean
print('검정통계량 t값:%.3f, p-value:%.3f'%result)
# 검정통계량 t값:0.346, p-value:0.747
# 해석 : p-value:0.747 > 0.05 이므로 귀무가설 채택 
#       표본평균과 모집단이 같다.
#       귀무가설 : 어느 남성 집단의 평큔키는 175.0다

print('참고: 평균키는 165.0이라고 한 경우')

result = stats.ttest_1samp(one_sample, popmean=165.0) # 이미 알려져있는 평균값:popmean
print('검정통계량 t값:%.3f, p-value:%.3f'%result)
# 검정통계량 t값:3.185, p-value:0.033
#       표본평균과 모집단이 같지 않다
# 해석 : p-value:0.033 < 0.05 이므로 귀무가설 기각, 상대가설 채택
#       상대가설 : 어느 남성 집단의 평균키는 175.0가 아니다.
print()

print("실습 예제 2")

# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균검정) student.csv

# 귀무가설 : A중학교 1학년 1반 학생들의 국어 점수의 평균은 80이 이다.
# 대립가설 : A중학교 1학년 1반 학생들의 국어 점수의 평균은 80이 아니다.

data =pd.read_csv("testdata/student.csv")
print(data.head(3))
print(data.describe())
print(data['국어'].mean()) # 72.9

result2 = stats.ttest_1samp(data['국어'], popmean=80.0)
print('검정통계량 t값:%.3f, p-value:%.3f'%result2)
# 검정통계량 t값:-1.332, p-value:0.199
# 해석 : p-value 0.199 >0.05 이므로 귀무가설 채택, 상대가설 기각
#       A중학교 1학년 1반 학생들의 국어 점수의 평균은 80이 이다.
print()

print("실습 예제3")

# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.
# gender : 여아 : 1 / 남아: 2

# 귀무가설 : 여아 신생아의 몸무게는 평균이 2800(g) 이다
# 대립가설 : 여아 신생아의 몸무게는 평균이 2800(g) 보다 더 크다

data =pd.read_csv("testdata/babyboom.csv")
print(data.head(3), len(data))
print()

fdata = data[data.gender == 1]
print(fdata, len(fdata))
print(np.mean(fdata.weight)) # 3132.4444444444443

# 정규성 검정 ( 두 집단 이상일때 주로 사용)
print(stats.shapiro(fdata.iloc[:,2]))
# ShapiroResult(statistic=0.8702831864356995, pvalue=0.017984945327043533)

# p-value(0.0179849) <0.05 이므로 정규성 만족x (0.05보다 크면 정규성 만족)

# histograme으로 정규분포 확인
sns.distplot(fdata.iloc[:, 2])
plt.show()
# 종의 모양으로 띄고있지만 대칭은 아니다.

stats.probplot(fdata.iloc[:, 2], plot=plt)
plt.show()

result3 = stats.ttest_1samp(fdata['weight'], popmean=2800.0)
print('검정통계량 t값:%.3f, p-value:%.3f'%result3)
# 검정통계량 t값:2.233, p-value:0.039
# 해석 : p-value 0.039 < 0.05 이므로 귀무가설 기각, 대립가설 채택
#       여아 신생아의 몸무게는 평균이 2800(g) 보다 더 크다

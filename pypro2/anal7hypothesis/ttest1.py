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
#       귀무가설 : 어느 남성 집단의 평큔키는 175.0다

print('참고: 평균키는 165.0이라고 한 경우')

result = stats.ttest_1samp(one_sample, popmean=165.0) # 이미 알려져있는 평균값:popmean
print('검정통계량 t값:%.3f, p-value:%.3f'%result)
# 검정통계량 t값:3.185, p-value:0.033
# 해석 : p-value:0.033 < 0.05 이므로 귀무가설 기각, 상대가설 채택
#       상대가설 : 어느 남성 집단의 평균키는 175.0가 아니다.
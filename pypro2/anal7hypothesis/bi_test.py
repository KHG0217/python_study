# 이항 검정 : 결과가 두가지 값을 가지는 확률변수의 분포 (이항 분포)를 판단하는데 효과적이다.
# stats.binom_test 함수를 사용

import pandas as pd
import scipy.stats as stats

# 직원 대상 고객안내 서비스 자료를 사용한다.
# 귀무가설 : 고객대응 교육 후 고객안내 서비스 만족율이 80% 이다.
# 대립가설 : 고객대응 교육 후 고객안내 서비스 만족율이 80% 가 아니다.
data =pd.read_csv('testdata/one_sample.csv')
print(data.head(3))
print(data['survey'].unique()) # [1 0]
ctab = pd.crosstab(index = data['survey'], columns='count')
ctab.index =['불만족','만족']
print(ctab)

print('양측 검정 : 방향성이 없다. ') # 크다 작다가 없다.
# 기존 80% 만족을 기준으로 검증 실시
x = stats.binom_test([136,14],p=0.8, alternative='two-sided')
print(x)
# 0.0006734701362867024 < 0.05 귀무 기각
# 대립가설 : 고객대응 교육 후 고객안내 서비스 만족율이 80% 가 아니다.
print()

# 기존 20% 불만족을 기준으로 검증 실시
x = stats.binom_test([14,136],p=0.2, alternative='two-sided')
print(x) 
print()
# 0.0006734701362867063 결과는 같음

print('단측 검정 : 방향성이 있다. ') # 크다 작다가 있다.
x = stats.binom_test([136,14],p=0.8, alternative='greater')
print(x)
# 0.00031794019219854805 <0.05 귀무기각
# 고객대응 교육 후 고객안내 서비스 만족율이 80% 보다 높다.(greater)
print()

x = stats.binom_test([14,136],p=0.2, alternative='less')
# 0.00031794019219854924 <0.05 귀무기각
# 고객대응 교육 후 고객안내 서비스 불만족율이 20% 보다 낮다.(greater)
print(x) 
print()


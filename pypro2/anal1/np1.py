# numpy :  ndarray를 지원한다. 파이썬의 데이터 관련 라이브러리 전체 생태계의 핵심을 지원한다.

# 직접 분산, 표준편차를 구하고, numpy의 함수와 비교.

grades = [1, 3, -2, 4] 

def show_grades(grades):
    for g in grades:
        print(g, end = ' ')
        
show_grades(grades)

def grades_sum(grades):
    tot = 0
    for g in grades:
        tot += g
    return tot
print()
print('합은 ', grades_sum(grades))

def grades_mean(grades):
    tot = grades_sum(grades)
    m = tot / len(grades)   # 합을 전체갯수로 나눠줌
    return m # 평균 (산술평균)
print('평균은 ', grades_mean(grades))

# 평균으로부터 떨어진 합 = 편차 = 원래값 - 평균
# 편차의 합은? 음수가 있기때문에 구할 수 없음.  -> 제곱을 씌워 계산 -> 전체합을 전체갯수로 나눔 -> 편차의 평균 = 분산
# 편차의합이 너무 클때? -> 루트씌워서 계산 -> 표준편차.

# 분산 : 평균값을 기준으로 다른 값들의 흩어짐 정도
def grades_variance(grades):    #분산 :variance
    m = grades_mean(grades)
    vari = 0
    for su in grades:
        vari += (su - m) ** 2 # 편차 제곱
    return vari / len(grades) # 파이썬은 모집단으로 나눔 계산 n / R은 자유도로 나눔 계산 n-1(n : 전체갯수)

print('분산은 ',grades_variance(grades))

# 표준편차: 분산에 루트를 씌운 값
def grades_std(grades):
    return grades_variance(grades) ** 0.5

print('표준편차는',grades_std(grades))

import numpy
print(numpy.__version__) # 1.21.5 버전
print('합은 ', numpy.sum(grades))
print('평균은 ', numpy.mean(grades))
print('분산은 ', numpy.var(grades))
print('표준편차는 ', numpy.std(grades))

print(numpy.average(grades)) # 가중평균도 구할 수있다.
print(numpy.mean(grades)) # 보통은 mean으로 평균을 구하면 된다.

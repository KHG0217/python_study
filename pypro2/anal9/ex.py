# import numpy as np

# 1
# data = np.array([[1,2,3,4],
#                         [5,6,7,8],
#                         [9,10,11,12],
#                         [13,14,15,16]])
# print(data)
# print(np.flip(data))

#2
# import seaborn as sns
# titanic = sns.load_dataset('titanic')
# print(titanic.head())
#
# print(titanic.pivot_table(index=titanic['sex'], columns=titanic['class']).iloc[:,21:24])
#
# from bs4 import BeautifulSoup
# import urllib.request as req
# import urllib

#3
# url = "https://news.v.daum.net/v/20220801160253605"
# daum = req.urlopen(url) # _blank
# convert_data = BeautifulSoup(daum, 'html.parser')
# for atag in convert_data.findAll('a', {'target':'_blank'}):
#     print(atag)

#4
# DataFrame의 행 열의 위치를 변경하기 위한 명령으로 빈칸 ①을 채우시오.
# 또한 DataFrame의 인덱스가 'd'인 행 삭제를 위한 명령을 빈칸 ②에 적으시오. (배점:5)
# from pandas import DataFrame
# frame = DataFrame({'bun':[1,2,3,4], 'irum':['aa','bb','cc','dd']}, index=['a','b', 'c','d'])
# print(frame.T)
#
# print()
# frame2 = frame.drop('d')
# print(frame2)

#5
# [문항5] "ex.csv" 파일을 읽어 칼럼명이 있는 Dataframe type의 자료를 얻으려고 한다.
# 판다스의 csv 파일 읽기 전용 함수를 사용하여 읽을 수 있도록 하자.
# 칼럼명은 a, b, c, d라고 하겠다. 아래의 빈칸에 적당한 소스 코드를 적으시오. (배점:5)
# ---- 현재 모듈과 같은 경로에 ex.csv 파일 내용 ----
# 1,2,3,4
# 5,6,7,8

# import pandas as pd
# # df = ①________________(②________ , ③____________)  # 칼럼명도 적어준다
#
# df=pd.read_csv("ex.csv",names=['a','b','c','d'])
# print(df)

#6
    
# [문항6] sqlite DB를 사용하여 DataFrame의 자료를 db에 저장하려 한다. 
# 아래의 빈칸에 알맞은 코드를 적으시오.
# 조건 : index는 저장에서 제외한다.
# (배점:5)
# import pandas as pd
# data = {
#     'product':['아메리카노','카페라떼','카페모카'],
#     'maker':['스벅','이디아','엔젤리너스'],
#     'price':[5000,5500,6000]
# }
#
# df = pd.DataFrame(data)
# df.to_sql('test', conn, if_exists='append', index=False)

#7

# [문항7] import matplotlib.pyplot as plt 
# 로 모듈을 임포트 했을 때 시각화 명령 중에서 실제로 차트를 출력하도록 하며,
# jupyter notebook에서 %matpltlib inline을 하면
#  생략이 가능한 함수명은 무엇인지 적으시오. (배점:5)

# .show()

#8

# [문항8] 데이터 분석 시 종속변수와 독립변수는 척도에 따라 분류방법이 달라진다.
# 아래의 빈칸을 채우시오. (배점:5)
# 독립변수    종속변수      분석방법
# 범주형      범주형        1)________________
# 범주형      연속형          T검정, ANOVA
# 연속형      범주형        2)_________________
# 연속형      연속형        3)_________________

# 1. 카이제곱 검정
# 2. 로지스틱 회귀분석
# 3. 회귀분석, 구조 방정식

#9
#
# [문항9] 기술 통계란 자료를 그래프나 숫자 등으로 요약하는 통계적 행위 및 관련 방법을 말한다.
# 중심 경향값(분포의 중심)을 표현하는 대표값으로 평균, 중위수, 최빈값 등이 있다.
# 그렇다면 산포도(분포의 퍼짐 정도)를 표현하는 측정치는 어떤 것들이 있는지 3가지 이상 적으시오. (배점:5)

# 평균편차, 분산, 표준편차

#10

#
# [문항10] 관찰된 빈도가 기대되는 빈도와 의미 있게 다른가(적합성, 독립성, 동질성)의 여부를 검정하기 위해 사용되는 가설검정 방법이다.
# 이 설명에 해당되는 검정 방법의 이름을 적으시오. (배점:5)
# 카이제곱검정

#11
#
# [문항11] 
# 크기가 다른 두 배열(numpy)을 작성하시오.
# x 변수에는 1차원 배열의 요소로 1 2 3 4 5를,
#  y 변수에는 2차원 배열(3행 1열)의 요소로 1 2 3을 저장한다.
# y 변수는 reshape 함수를 사용한다.
# 두 배열 간 더하기(+) 연산을 하면 아래와 같은 결과가 나오는데
#  그 이유도 간단히 적으시오.

# 연산결과
# [[2 3 4 5 6]
# [3 4 5 6 7]
# [4 5 6 7 8]] (배점:10)
# import numpy as np
# x = np.array([1,2,3,4,5])
# y = np.arange(1,4).reshape(3,1)
#
# 연산결과 이유:
# 크기가 다른 배열간의 연산을 할 경우 부족한 열과행을
# 채워 계산하는 Broadcasting 연산을 하기 떄문이다.

#12

# [문항12] 네이버 사이트가 제공하는 실시간 인기 검색어 자료를 읽어, 
# 사람들에게 관심 있는 주제는 무엇인지 알아보려 한다.
# title을 얻기 위해 ol tag 내의 li tag 값을 얻기 위한 코드를 아래와 같이 작성하였다.
# 프로그램이 제대로 수행될 수 있도록 아래의 빈 칸을 채우시오. (배점:10)

 
# from bs4 import BeautifulSoup
# import urllib.request      
# try:
#     url = "http://www.naver.com"
#     page = urllib.request.urlopen(url)
#
#     soup = BeautifulSoup(page.read(), "html.parser") 
#     title = soup.select_one('ol').find_all('li')
#     print(title)
#     for i in range(0, 10):
#             print(str(i + 1) + ") " + title[i].a['title'])
# except Exception as e:
#     print('에러:', e)



# try:
#       url = "http://www.naver.com"
#       page = urllib.request.urlopen(url)
#
#       soup = BeautifulSoup(page.read(), "1)___________") 
#       title = soup.2)_____.find_all('li')
#       for i in range(0, 10):
#               print(str(i + 1) + ") " + title[i].a['title'])
# 3)_________ Exception as e:
#       print('에러:', e)
# 1.html.parser
# 2.select_one('ol')
# 3. except


# [문항13] DataFrame 객체 타입의 데이터를 "test.csv" 파일로 저장하려 한다.
# index와 header는 저장 작업에서 제외한다.
# 아래의 소스 코드를 순서대로 완성하시오. (배점:10)
# data = DataFrame(items)
# data.to_csv(              , index=        , header=          )
# from pandas import DataFrame

# items=[1,2,3,4,5,6,7,8,9]
# data = DataFrame(items)
# data.to_csv("test.csv",index=  False , header= False )


    
# [문항14] 다음 코드를 참조해서 내용에 맞는 소스 코드를 적으시오.
#  pandas 모듈을 이용한다.
#
#
# data = {"a": [80, 90, 70, 30], "b": [90, 70, 60, 40], "c": [90, 60, 80, 70]}
#
# 칼럼(열)의 이름을 순서대로 "국어", "영어", "수학"으로 변경한다.
# 아래 문제는 제시한 columns와 index 명을 사용한다.
# 1) 모든 학생의 수학 점수를 출력하기
# 2) 모든 학생의 수학 점수의 표준편차를 출력하기
# 3) 모든 학생의 국어와 영어 점수를 Series 타입이 아니라 DataFrame type으로 출력하기 (배점:10)

# from pandas import DataFrame,Series
# data = {"a": [80, 90, 70, 30], "b": [90, 70, 60, 40], "c": [90, 60, 80, 70]}
#
# frame = DataFrame(data)
# frame.columns = ['국어', '영어','수학']
# print(frame)
# print(frame.수학)
# print(frame.수학.std())
# data1 = frame.국어.to_frame()
# data2 = frame.영어.to_frame()
# print(frame.국어.to_frame())
# print(frame.영어.to_frame())

# [문항15] 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다.
# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 two-sample t-검정을 하시오.

# blue : 70 68 82 78 72 68 67 68 88 60 80
# red : 60 65 55 58 67 59 61 68 77 66 66 (배점:10)
# ① 귀무가설 : 포장지 색상에 따른 제품의 매출액에 차이가 존재하지 않는다.
# ② 대립가설 : 포장지 색상에 따른 제품의 매출액에 차이가 존재한다.
# ③ 검정을 위한 소스 코드




# import scipy.stats as stats
# blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
# red =  [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]
# print(blue)
# result1 = stats.levene(blue,red)
# print(result1)
# #pvalue=0.4391644468508382 >p벨류가 0.05보다 이므로 대립가설 기각 귀무가설 채택 


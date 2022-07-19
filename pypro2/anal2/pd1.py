# Pandas : 고수준의 자료구조와 빠르고 쉬운 분석용 자료구조 및 함수를 지원
# data munging or data wrangling: DATA를 가공하는 과정 / 작업을 효율적으로 처리 가능 
# -> 원자료(raw data)를 보다 쉽게 접근하고 분석할 수 있도록 데이터를 정리하고 통합하는 과정

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from bokeh.layouts import column

# Series : 일련의 자료를 담을 수 있는 1차원 배열과 유사한 자료구조로 색인을 갖는다. 색인 = 인덱싱

obj  = pd.Series([3,7,-5,4]) # list 가능
# obj  = pd.Series((3,7,-5,4)) # tuple 가능
# obj  = pd.Series({3,7,-5,4}) # set type은 x 'set' type is unordered
print(obj, type(obj)) #<class 'pandas.core.series.Series'> class의 포함관계 

obj2  = pd.Series([3,7,-5,4], index=['a','b','c','d']) # 인덱싱을 직접 줄 수도 있음
print(obj2)
print(obj2.sum(),sum(obj2), np.sum(obj2)) # 합 구하기
print(obj2.mean(), obj2.std()) # 평균, 표준편차

print()
print(obj2.values) # , 가 없으므로 array인것을 알 수있다.
print(obj2.index) # index 보기

print('------------------슬라이싱--------------------------')
print(obj2['a'], ' ', obj2[['a']]) # a값:3 / 인덱스값 / a의 인덱스 값:3
print(obj2[0]) 
print(obj2[['a','b']]) #색인명 두개이상 찾기 [[]] !
print(obj2['a':'c']) # 색인명 슬라이싱
print(obj2[1:4])
print(obj2[[2,1]])
print(obj2 > 1) # 값이 1 보다 큰가? True or False
print('a' in obj2) # a 가 obj2에 들어가 있는가? True or False

print('dict type으로 Series 객체 생성 ---')
names = {'mouse':5000,'keyboard':35000,'moniter':550000}
obj3 =Series(names)
print(obj3, type(obj3))
obj3.index = ['마우스','키보드','모니터'] # index 바꾸기
print(obj3)
print(obj3['마우스'])
print(obj3[0])

obj3.name = '상품가격'
print(obj3) # Series 객체 자체에 이름을 주기.

print('----DataFrame : 표 모양의 자료구조 ----------')
df = DataFrame(obj3) # obj3가 하나의 칼럼으로 들어갔다.
print(df, type(df)) 

data = {
    'irum':['홍길동', '한국인', '신기해', '공기밥', '한가해'],
    'juso':('역삼동', '신당동', '역삼동', '역삼동', '신사동'),
    'nai':[23, 25, 33, 30, 35],
}
print(data, type(data)) # dict 타입
frame = DataFrame(data) # Series들이 모여서 만들어진 Frame
print(frame)
print(frame.irum,type(frame.irum)) # frame.irum = Series = 각각의 칼럼들은 Series다.
print(frame['irum'])

print(DataFrame(data, columns = ['juso','nai','irum'])) # 칼럼순서 바꾸기

print()
frame2 = DataFrame(data, columns = ['irum','juso','nai','tel'], 
                   index=['a','b','c','d','e'] ) # 칼럼 추가도 가능, index 설정도 가능

print(frame2)
frame2['tel'] = '111-1111'
print(frame2)

val = Series(['222-1111','333-1111','444-1111'], index=['b','c','e'])
print(val)
frame2['tel'] = val # 덮어쓰기가 됨, 처음에 넣은 111-1111은 사라짐
print(frame2)

print()
print(frame2.T) # 행 열을 바꿔줌

print(frame2.values) # 메트릭스로 값으로 돌려주고 있다.
print(frame2.values[0, 1]) # 23
print(frame2.values[0:2]) # 0행부터 1행까지
print(type(frame2.values[0:2])) # numpy.ndarray

print('행 또는 열 삭제') # axis =0 행, axis = 1 열
frame3 = frame2.drop('d') # axis = 0 생략 = d행을 삭제
print(frame3)

frame4 = frame2.drop('tel',axis=1) # axis = 1 열삭제 = tel 열을 삭제
print(frame4)

print('정렬 ---------')
print(frame2.sort_index(axis = 0, ascending=False)) # 행을 내림차순
print(frame2.sort_index(axis = 1, ascending=True)) # 열을 내림순
print(frame2.rank(axis = 0)) # 순위 행방향으로 등수를 매김

counts =frame2['juso'].value_counts() # 'juso'칼럼에 대한 빈도수 구하기
print('칼럼값 갯수: ', counts)

print(' 문자열 자르기 --------')
data = {
    'juso':['강남구 역삼동','중구 신당동','강남구 대치동'],
    'inwon':[23, 25, 15],
    
}

fr = DataFrame(data)
print(fr)

result1 = Series([x.split()[0] for x in fr.juso]) # List로 담기 
result2 = Series((x.split()[1] for x in fr.juso)) # Tuple로 담기
print(result1)
print(result2)
print(result1.value_counts()) # split()을 기준으로 [0]번째 = 구 의 빈도수 구하기

# --------------------------------------------
print('Series의 재색인(Reindex)')
data = Series([1,3,2], index=(1,4,2)) # index 부분은 set도 가능하다.
print(data)

data2 = data.reindex((1,2,4)) # data 인덱스 다시 설정하기.
print(data2) 

print('재색인 시 값 채워넣기')
data3 = data2.reindex([0, 1, 2, 3, 4, 5]) # 대응값이 없는 index는 NaN(결측치)
print(data3)

data3 = data2.reindex([0, 1, 2, 3, 4, 5], fill_value = 77) # data2 에 대응값이 없으면 임의이 특정값(7)을 넣어줌
print(data3)

# data4 = data2.reindex([0, 1, 2, 3, 4, 5], method = 'ffill')  # data2에 대응값이 없으면 바로 앞 index값을 채워 넣어줌
data4 = data2.reindex([0, 1, 2, 3, 4, 5], method = 'pad')
print(data4)

# data5= data2.reindex([0, 1, 2, 3, 4, 5], method = 'bfill') # data2에 대응값이 없으면 바로 뒤 index값을 채워줌
data5= data2.reindex([0, 1, 2, 3, 4, 5], method = 'backfill')
print(data5)

print('bool 처리 ----------')
df = DataFrame(np.arange(12).reshape(4,3),
               index = ['1월','2월','3월','4월'], columns=['강남','강북','서초']) 
print(df)
print(df['강남'] > 3)
print(df[df['강남'] > 3]) # 조건이 True 행만 출력

print()
print(df < 3)
df[df < 3] = 0 # 3 보다 작은 값은 0으로 대체
print(df)

print('DataFrame 관련 슬라이싱 함수 : loc() - 라벨 지원 , iloc() - 순서(숫자) 지원')
print(df.loc['3월', :]) # 행과 열 ('3월'행, 모든열) / 반환값 dataframe
print(df.loc['3월', ]) # 행과 열 ('3월'행, 모든열) / : 생략 가능 / 반환값 dataframe
print(df.loc[:'2월',['서초']]) # 행과 열 ('2월'이하행, 서초열) / 반환값 dataframe
print()

print(df.iloc[2]) # 2행 전부
print(df.iloc[2, :])# 2행 전부

print(df.iloc[:3]) # 3행 미만
print(df.iloc[:3, 2]) # 3행 미만, 2열(0~3)
print(df.iloc[1:3, 1:3]) # 1,2행 ㅣ만, 1,2열 반환







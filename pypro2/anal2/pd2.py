# 연산 
from pandas import Series, DataFrame
import numpy as np
from django.contrib.gis.db.backends import oracle

s1 = Series([1,2,3], index=['a','b','c'])
s2 = Series([4,5,6,7], index=['a','b','d','c'])
print(s1) 
print(s2)
# 인덱스가 맞지않는 (여기선 d)는 NaN값이 들어간다
print(s1 + s2) # Series 객체간 더하기
# print(s1.add(s2)) # .add 함수 사용

print(s1 - s2) # Series 객체간 빼기

print(s1 * s2) # Series 객체간 곱하기

print(s1 / s2) # Series 객체간 나누기

print() # DataFrame 객체 간 연산
df1 = DataFrame(np.arange(9.).reshape(3,3), 
                columns=list('kbs'), index=['서울','대전','부산'])
# 칼럼에  k b s로 칼럼이 각각 들어감 (총3개)


df2 = DataFrame(np.arange(12.).reshape(4,3), 
                columns=list('kbs'), index=['서울','대전','제주','수원'])
print(df1)
print(df2)
print(df1 + df2) # 대응되는 곳이 없으면 NaN 값이 들어감 + - * /
print(df1.add(df2, fill_value = 0)) # NaN을 0으로 채운후 연산에 참여하기 때문에 ->  원래값도 그냥 들어감
# add, sub, mul, div

seri = df1.iloc[0]
print(seri)
print(df1)
print()
 
print(df1+seri) # DataFrame/Series 연산 : Broadcasting

# 기술 통계 관련 함수 : 수집한 데이터를 요약, 묘사, 설명하는 통계 기법
print('결측 값 처리 ')
df = DataFrame([[1.4, np.nan],[7, -1.5],[np.NaN, np.NAN],[0.5, -1]],columns=['one','two'])
print(df)
print()

print(df.drop(1)) # 1행 (특정) 삭제

print(df.isnull()) # null 이면 True
print(df.notnull()) # notnull 이면 True

print(df.dropna()) # na가 하나라도 들어있으면 지운다.

print(df.dropna(how='any')) # NaN이 하나라도 들어있으면 해당 행 삭제

print(df.dropna(how='all')) # 모든 값이 NaN인 경우 해당 행 삭제

print(df.dropna(subset=['one'])) # 'one'칼럼(열)에 NaN이 있으면 해당 행 삭제

print(df.dropna(axis='rows')) # NaN이 포함된 행의 해당 행 지우기

print(df.dropna(axis='columns')) # NaN이 포함된 열의 해당 행 지우기

print()
print(df.fillna(0)) # NaN 값을 0으로 채우기

print(df.fillna(method='ffill')) # NaN을 바로 앞 인덱스값으로

print(df.fillna(method='bfill')) # NaN을 바로 뒤 인덱스값으로

print('내장함수 ')
print(df)

print(df.sum()) # 열의 합
print(df.sum(axis=0))
print()

print(df.sum(axis=1)) # 행의 합
print(df.mean(axis=1)) # 행의 평균
print(df.mean(axis=1, skipna=False)) # 행의 평균 / NaN를 연산에서 제외

print()

print(df.describe()) # 요약 통계량을 출력
print(df.info()) # 구조 출력

words = Series(['봄','여름','봄'])
print(words.describe())
# print(words.info()) # Series의 info는 볼게 없다.

 

 


# * Pandas의 DataFrame 관련 연습문제 *
#
# pandas 문제 1)
#
#   a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오. 
#
#      np.random.randn(9, 4)
#
#   b) a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오
#
#
#
#   c) 각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용

data = np.random.randn(9, 4)
fr = DataFrame(data)

print(fr)

fr = DataFrame(data, columns=['No1','No2','No3','No4'])
print(fr)

print(fr.mean(axis=0)) # 칼럼이니까 행

# 문제 2

data2 = {
        'numbers':[10,20,30,40]
    }
fr2 = DataFrame(data2,columns=['numbers']
                ,index=['a','b','c','d'])
print(fr2)

print(fr2.loc['c'])
print(fr2.loc[['a','d']])
print(fr2.sum())
print(fr2.loc['a':]**2)

fr2['floats'] = [1.5, 2.5,3.5,4.5]
fr2['names'] = Series(['길동','오정','팔계','오공'], index=['d','a','b','c'])
print(fr2)

# fr2 = DataFrame(data2,columns=['numbers','floats','names']
#                 ,index=['a','b','c','d'])
#
# floats = Series([1.5,2.5,3.5,4.5], index=['a','b','c','d'])
# names = Series(['길동','오정','팔계','오공'], index=['d','a','b','c'])
#
#
# fr2['floats']=floats
# fr2['names']=names
# print(fr2)














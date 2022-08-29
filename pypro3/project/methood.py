
s = 'sequence'
print(len(s), s.count('e'))
print(s.find('e'), s.find('e', 3), s.rfind('e'))


# numpy
import numpy
import numpy as np
b = np.array([[1,2,3],[4,5,6]])
print(numpy.__version__) # 1.21.5 버전
print('합은 ', numpy.sum(b))
print('평균은 ', numpy.mean(b))
print('분산은 ', numpy.var(b))
print('표준편차는 ', numpy.std(b))
print(b[[0]].flatten()) # 다차원을 1차원으로 바꾸는 함수
print(b[[0]].ravel()) # 다차원을 1차원으로 바꾸는 함수

a = np.array([1,2,3])

a = np.array([1,2,3,4,5])
print(a[1])
print(a[1:5:2])
print(a[1:])
print(a[-2:])

b = a # 주소를 치환
b[0] = 77
print(a)
print(b)
del b

c = np.copy(a) # 복사본을 만드는 것.
c[0] = 88
print(a) # b에서 바꾼값이 a 에 들어감 -> 주소를 같이 쓰기 떄문
print(c)
del c
print(a)

print()
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a)
print(a[0]) # 벡터 [1 2 3 4]
print(a[0,0]) # 스칼라 1
print(a[[0]]) # 매트릭스 [[1 2 3 4]]
print(a[1:, 0:2]) # 1행 이후로 0열과 1열만 



# Pandas
import pandas as pd
from pandas import Series, DataFrame
print('------------------슬라이싱--------------------------')
obj2  = pd.Series([3,7,-5,4], index=['a','b','c','d']) # 인덱싱을 직접 줄 수도 있음
print(obj2['a'], ' ', obj2[['a']]) # a값:3 / 인덱스값 / a의 인덱스 값:3
print(obj2[0]) 
print(obj2[['a','b']]) #색인명 두개이상 찾기 [[]] !
print(obj2['a':'c']) # 색인명 슬라이싱
print(obj2[1:4])
print(obj2[[2,1]])
print(obj2 > 1) # 값이 1 보다 큰가? True or False
print('a' in obj2) # a 가 obj2에 들어가 있는가? True or False

names = {'mouse':5000,'keyboard':35000,'moniter':550000}
obj3 =Series(names)
obj3.index = ['마우스','키보드','모니터'] # index 바꾸기

print('----DataFrame : 표 모양의 자료구조 ----------')
df = DataFrame(obj3) # obj3가 하나의 칼럼으로 들어갔다.
print(df, type(df))

df = DataFrame(np.arange(12).reshape(4,3),
               index = ['1월','2월','3월','4월'], columns=['강남','강북','서초']) 

print('DataFrame 관련 슬라이싱 함수 : loc() - 라벨 지원 , iloc() - 순서(숫자) 지원')
print(df.loc['3월', :]) # 행과 열 ('3월'행, 모든열) / 반환값 dataframe
print(df.loc['3월', ]) # 행과 열 ('3월'행, 모든열) / : 생략 가능 / 반환값 dataframe
print(df.loc[:'2월',['서초']]) # 행과 열 ('2월'이하행, 서초열) / 반환값 dataframe
print()


print('lioc')
print(df.iloc[2]) # 3행 전부 [0]시작
print(df.iloc[2, :])# 3행 전부
print()

print(df.iloc[:3]) # 4행 미만
print(df.iloc[:3, 2]) # 4행 미만, 3열(0~3)
print(df.iloc[1:3, 1:3]) # 2,3행 미만, 2,3열 반환

# DataFrame 객체 병합 : merge 
df1 = pd.DataFrame({'data1':range(7), 'key':['b','b','a','c','a','a','b']})
print(df1)
df2 = pd.DataFrame({'key':['a','b','d'],'data2':range(3)})
print(df2)
print('inner ------')

print(pd.merge(df1, df2, on ='key')) # merge(기준치,기준치, on=기준값)/ key 를 기준으로 병합 (inner join : 교집합)
# key 중에서 공통으로 가지고 있는 'a' 와 'b'만 가져온다.
print()

print(pd.merge(df1, df2, on ='key', how ='inner' )) # how 병합방법 / 기본값 inner

print('outer ------')
print(pd.merge(df1, df2, on ='key', how ='outer' )) # key 를 기준으로 병합 (full outer join)
# key 값이 다나옴 'a','b','c','d'

print('left ------')
print(pd.merge(df1, df2, on ='key', how ='left' )) # key 를 기준으로 병합 (left outer join)
# left 여서 df1값은 다나옴 'a','b','c'

print('right ------')
print(pd.merge(df1, df2, on ='key', how ='right' )) # key 를 기준으로 병합 (right outer join)
# right 여서 df2값은 다나옴 'a','b','d'

print('공통 칼러명이 없는 경우 -------------')
df3 = pd.DataFrame({'key2':['a','b','d'],'data2':range(3)})
print(df3)
print(df1)
# df3와 df1은 칼럼명이 같지않음 .
print()

print(pd.merge(df1, df3, left_on ='key',right_on ='key2', how ='inner' ))
# 왼쪽은 key를 오른쪽은 key2로 inner 조인

print('자료 이어 붙이기') 
print(pd.concat([df1,df3], axis=0)) # 조인값 없이 그냥 이어 붙이기 행으로 ?
print()
print(pd.concat([df1,df3], axis=1)) # 조인값 없이 그냥 이어 붙이기 열로 ?

print('피봇(pivot) ------------------')
# 열을 기준으로 구조를 변경하여 새로운 집계표를 작성
data = {'city':['강남','강북','강남','강북'],
        'year':[2000,2001,2002,2002],
        'pop':[3.3,2.5,3.0,2.0]}
df = pd.DataFrame(data)
print(df)

print('privot------------------')
print(df.pivot('city', 'year', 'pop'))
# city가 행, year가 칼럼, pop가 벨류
print()

print(df.set_index(['city','year']).unstack())
# set_index : 기존의 행 인덱스를 제거하고, 첫번째 열 인덱스를 설정
# pivot과 같은 형태가 됨

print('groupby------------------') 
hap = df.groupby(['city'])
print(hap.sum())
print(df.groupby(['city']).sum()) # 위 두줄을 한 줄로 표현

print(df.groupby(['city','year']).mean()) # city 별 year별 pop 평균

print()
print(df.groupby(['city']).agg('sum'))

print()
print(df.groupby(['city','year']).agg('sum'))

print()
print(df.groupby(['city','year']).agg('mean'))

print()
print(df.groupby(['city','year']).agg(['mean','std']))

print('pivot _table-------------')
print(df)

print(df.pivot_table(index=['city'])) # 평균 계산 / 평균이 기본

print(df.pivot_table(index=['city'], aggfunc=np.mean)) # 위와 같은 값 # 결과는 year,pop값

print(df.pivot_table(index=['city','year'], aggfunc=[len,np.sum])) # 갯수와 합 #결과는 pop값

print(df.pivot_table(values=['pop'],index=['city'])) # city별 pop의 평균
print(df.pivot_table(values=['pop'],index=['city'], aggfunc=np.mean))
print(df.pivot_table(values=['pop'],index=['city'], aggfunc=len))
print(df.pivot_table(values=['pop'],index=['year'], columns=['city']))
print(df.pivot_table(values=['pop'],index=['year'], columns=['city'],
                     margins=True))
print(df.pivot_table(values=['pop'],index=['year'], columns=['city'],
                     margins=True, fill_value=0))



# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family = 'malgun gothic') # 그래프에 한글 깨질 때
plt.rcParams['axes.unicode_minus'] = False # 한글깨짐 방지후 음수깨짐 방지

# 차트를 이미지로 저장하기
fig = plt.gcf()
plt.show()
fig.savefig('test.png')

# 이미지 파일 읽기
from matplotlib.pyplot import imread
img = imread('test.png')
plt.imshow(img)
plt.show()

# 차트 종류


'''
x = np.arange(10)
y = np.sin(x)
z = np.cos(x)

# 차트 영역 객체 선언 방법 1
plt.figure()    # matplotlib 스타일의 인터페이스 
plt.subplot(2,1,1) # row ,cilumn, panel number
plt.plot(x, y)
plt.subplot(2,1,2)
plt.plot(x, z)
plt.show()

# 차트 영역 객체 선언 방법 1-1
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.hist(np.random.randn(10), bins = 5, alpha = 0.3) # alpha 투명도
ax2.plot(np.random.randn(10))
plt.show()

# 차트 영역 객체 선언 방법 2
fig, ax = plt.subplots(nrows=2, ncols=1) # 2행 1열짜리 / 객체지향 인터페이스
ax[0].plot(x, np.sin(x))
ax[1].plot(x ,np.cos(x))
plt.show()
'''


data = [50, 80, 100, 70, 90]
'''
# plt.bar(range(len(data)), data) # 세로막대
plt.barh(range(len(data)), data) # 가로막대
plt.show()


plt.pie(data,explode=(0,0.1,0,0,0.5), colors=['yellow', 'red', 'blue']) #pie 형태 /explode 튀어나온 정도
plt.show()


plt.boxplot(data) # 값이 너무 크거나,낮은 하나의 특정데이터 확인가능
plt.show()


plt.scatter(data,data) # 산점도
plt.show()


# DataFrame 자료로 그래프 그리기
import pandas as pd
fdata = pd.DataFrame(np.random.randn(1000, 4),
                     index=pd.date_range('1/1/2000',periods=1000),
                     columns=list('ABCD'))

fdata = fdata.cumsum()
plt.plot(fdata)
plt.show()

# pandas의 plot 기능
print(type(fdata)) # <class 'pandas.core.frame.DataFrame'>
fdata.plot(kind= 'box')
plt.xlabel('time')
plt.ylabel('data')
plt.show()
'''

# seaborn 모듈 : matplotlib의 기능을 추가한 시각화 package
import seaborn as sns

titanic = sns.load_dataset('titanic')
print(titanic.info())

# sns.displot(titanic['age'])
# sns.boxplot(y='age', data=titanic)
t_pivot = titanic.pivot_table(index='class',columns='sex',aggfunc='size') # 색이 진할수록 밀도가 높은것
sns.heatmap(t_pivot)
plt.show()





# file i/o

import os

print(os.getcwd()) #현재 모듈의 경로

try:
    print('파일 읽기')
    # f1 = open(r'C:\Users\acorn\Desktop\GitRepository\python_study\pypro1\pack2\ftest.txt', mode='r' , encoding='utf-8') #이스케이프 문자 안읽기 위해 r
    # f1 = open(os.getcwd() + r'\ftest.txt', mode='r' , encoding='utf-8')
    f1 = open('ftest.txt', mode='r' , encoding='utf-8') # 현재경로일때는 경로를 생략해도 된다.
    print(f1.read())
    f1.close() # 메모리 효율적인 관리를 위해 close로 닫아주기
    
    print('파일 저장')
    f2 = open('ftest2.txt', mode='w', encoding='utf-8')
    f2.write('손오공\n')
    f2.write('사오정\n')
    f2.write('저팔계\n')
    f2.close()
    
    print('파일 추가')
    f2 = open('ftest2.txt', mode='a', encoding='utf-8')
    f2.write('김치국\n')
    f2.write('공기밥\n')
    f2.close()    

except Exception as e:
    print('err : ', e)
    
print('with 구문을 사용하면 close() 자동 처리 ----')
try:
    #저장
    with open('ftest4.txt', mode='w', encoding='utf=8') as obj1: # with 을 쓰면 close()를 안해도 자동으로 닫아준다. # mode = 생략가능
        obj1.write('파이썬으로 파일 처리\n')
        obj1.write('with 처리\n')
        obj1.write('close 생략\n')
        
    # 읽기
    with open('ftest3.txt', 'r', encoding='utf=8') as obj2: 
        print(obj2.read())

        
except Exception as e2:
    print('err2 : ', e2)

print()    
print('피클일(객체를 파일로 저장 및 읽기) ----')
import pickle

try:
    #개체 저장
    dicData = {'tom':'111-1111', 'john':'222-2222'}
    listData = ['장마철', '장대비 예고']
    tupleData = (dicData, listData)
    
    with open('hello.data', mode='wb') as ob1:  #hello.data파일을 만들고 tupleData,listData의 내용을 넣음
        pickle.dump(tupleData, ob1)   # pickle.dump(대상, 파일객체)
        pickle.dump(listData, ob1)
        
    #객체 읽기
    with open('hello.data', mode='rb') as ob2:
        a, b = pickle.load(ob2) # 먼저저장한 순서대로 a,b에 들어감
        print(a)
        print(b)
        print()
        c =  pickle.load(ob2)
        print(c)
            
except Exception as e3:
    print('err3 : ', e3)    
    
    
# 원격 데이터버베이스 서버(MariaDB)와 연동
# pip install mysqlclient 로 드라이브 파일 설치 (아나콘다 프롬프트)

import MySQLdb
"""
conn = MySQLdb.connect(host = '127.0.0.1', user = 'root', port = 3306,#port = 기본값 3306(생략가능) 기본 3306이라면 생략가능
                       password='maria123', database='test') #mariadb에 접속하기, dict타입

print(conn)
conn.close()
"""
config = {

    'host':'127.0.0.1',

    'user':'root',

    'password':'maria123',

    'database':'test',

    'port':3306,

    'charset':'utf8',

    'use_unicode':True

} #dict 타입 

try:
    conn = MySQLdb.connect(**config) #아규먼트에 dict 타입을 원하므로 **
    cursor = conn.cursor()
    
    #자료 추가
    # sql = "insert into sangdata(code,sang,su,dan) values(10,'상품1',5,1000)"
    # cursor.execute(sql)
    # conn.commit() # 파이썬은 반드시 커밋을 해줘야한다. !java는 오토커밋
                #?,?,?,? 자리에 %s
    """ 
    sql = "insert into sangdata values(%s,%s,%s,%s)"
    sql_data = ('11','상품2',12,2000) #tuple로 보내줌 () 안써도 tuple
    cou = cursor.execute(sql,sql_data) #sql 에 sql_data를 1:1로 맵핑해줌
    conn.commit()
    print('cou :', cou)
    if cou ==1:
        print('추가성공')
    """
    """
    # 자료 수정
    sql = "update sangdata set sang=%s,su=%s,dan=%s where code=%s"
    sql_data=('파이썬',7,5000,10)
    cursor.execute(sql,sql_data)
    conn.commit()
    """
    """
    # 자료 삭제
    code = '10'
    
    #비권장 secure coding guideline에 위배 sql injection 해킹 위험
    # sql = "delete from sangdata where code=" + code 
    
    #sql = "delete from sangdata where code='{0}".format(code) # 권장 1
    sql = "delete from sangdata where code=%s" #권장 2
    cursor.execute(sql,(code,)) #(code,) <-반드시 tuple형식으로 넣어야한다
    conn.commit()
    """
    
    #자료 읽기
    sql = "select code, sang, su, dan from sangdata"
    cursor.execute(sql)
    
    for data in cursor.fetchall():
        # print(data)
        print('%s %s %s %s'%data)
        
    """    
    print()
    for r in cursor:
        # print(r)
        print(r[0], r[1], r[2], r[3])
        
    print()
    for (code, sang, su, dan) in cursor:
        print(code, sang, su, dan)
    
    print()
    for (a, b, c, d) in cursor:
        print(a, b, c, d)
    """
except Exception as e:
    print('에러: ',e)
finally:
    cursor.close()
    conn.close()
    
# 분석

# 세 개 이상의 모집단에 대한 가설검정 – 분산분석

# 분산분석의 전제조건
# - 독립성 : 각 집단은 서로 독립
# - 정규성 : 각 집단은 정규분포를 따른다.
# - 불편성 : 등분산성을 갖춰야 함. (편향된 데이터가 아니여야 한다.) 

# anova

# 교호(상호) 작용 처리함
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import urllib.request

url = "https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3_2.txt"
data = pd.read_csv(urllib.request.urlopen(url))
print(data)
ols1 = ols("머리둘레 ~ C(태아수) + C(관측자수) +C(태아수):C(관측자수)", data = data).fit()
result = sm.stats.anova_lm(ols1, typ = 2)# 동질성 검정 - 두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법이다. 두 집단 이상에서 각 범주(집단) 간의 비율이 서로
print(result)
print()
# 교호작용이 들어갔을때 그 p-value 값을 읽어 해석한다
#C(태아수):C(관측자수)    0.562222   6.0     1.222222  3.295509e-01


# - -----------------------------------------
# 해석 : p-value : 3.295509e-01 > 0.05 이므로 귀무가설 채택

# 이원카이제곱 - 교차분할표 이용
# : 두 개 이상의 변인(집단 또는 범주)을 대상으로 검정을 수행한다.
# 분석대상의 집단 수에 의해서 독립성 검정과 동질성 검정으로 나뉜다.

# 독립성(관련성) 검정
# - 동일 집단의 두 변인(학력수준과 대학진학 여부)을 대상으로 관련성이 있는가 없는가?
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다.
# 실습 : 교육수준과 흡연율 간의 관련성 분석 : smoke.csv'

# 귀무가설 : 교육수준과 흡연율 간의 관령성이 없다. (독립이다.)
# 대립가설 : 교육수준과 흡연율 간의 관령성이 있다. (독립이 아니다.)

# 해석 : p -vaule 값이 < 0.05(유의수준) 이므로 귀무가설 기각 대립가설 채택
#       교육수준과 흡연율 간의 관령성이 있다.(독립이 아니다.)

# 동질성 검정 - 두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법이다. 두 집단 이상에서 각 범주(집단) 간의 비율이 서로
# 동일한가를 검정하게 된다. 두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법이다.


# 집단 간 차이분석: 평균 또는 비율 차이를 분석
# : 모집단에서 추출한 표본정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있다.
# * T-test와 ANOVA의 차이
# - 두 집단 이하의 변수에 대한 평균차이를 검정할 경우 T-test를 사용하여 검정통계량 T값을 구해 가설검정을 한다.
# - 세 집단 이상의 변수에 대한 평균차이를 검정할 경우에는 ANOVA를 이용하여 검정통계량 F값을 구해 가설검정을 한다.


# 상관 관계
# 공분산 print(np.cov(data.친밀도, data.적절성))
# 상관 계수 print(np.corrcoef(data.친밀도, data.적절성))

# 시각화
# import seaborn as sns
# sns.heatmap(data.corr())
# plt.show()

# 다른시각화 사용하기
# heatmap에 텍스트 표시 추가사항 적용해 보기
# corr = data.corr()
# # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)  # 상관계수값 표시
# mask[np.triu_indices_from(mask)] = True
# # Draw the heatmap with the mask and correct aspect ratio
# vmax = np.abs(corr.values[~mask]).max()
# fig, ax = plt.subplots()     # Set up the matplotlib figure
#
# sns.heatmap(corr, mask=mask, vmin=-vmax, vmax=vmax, square=True, linecolor="lightgray", linewidths=1, ax=ax)
#
# for i in range(len(corr)):
#     ax.text(i + 0.5, len(corr) - (i + 0.5), corr.columns[i], ha="center", va="center", rotation=45)
#     for j in range(i + 1, len(corr)):
#         s = "{:.3f}".format(corr.values[i, j])
#         ax.text(j + 0.5, len(corr) - (i + 0.5), s, ha="center", va="center")
# ax.axis("off")
# plt.show()

## 선형회귀식 얻기
# 최소제곱법으로 최적의 추세선을 구할 수 있는 기울기와 절편을 얻는다.
# 직접 수식을 쓸 수 있으나 numpy의 최소제곱(자승)해를 얻는 함수 사용.

print('# 회귀분석모형의 적절성을 위한 선행 조건도 체크 ---')
fitted = model.predict(df.iloc[:, [0,2,3,5,6]])
residual = df['Sales'] - fitted  # 잔차
print('residual : ', residual)

print('선형성 ---')
sns.regplot(fitted, residual, lowess = True, line_kws = {'color':'red'})
plt.plot([fitted.min(), fitted.max()], [ 0, 0], '--', color='blue')
plt.show()  # 미흡하지만 선형성을 만족

print('정규성 ---')
sr = scipy.stats.zscore(residual)
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x, y)
plt.plot([-3, 3],[-3, 3], '--', color='green')
plt.show()

# print('잔차의 정규성 : ', scipy.stats.shapiro(residual))
# # ShapiroResult(statistic=0.994922399520874, pvalue=0.2127407342195511)
# # pvalue=0.2127407342195511 > 0.05 # 정규성을 만족
#
# print('독립성 ---')
# # Durbin-Watson : 1.931  - 0 ~ 4 사이의 값을 갖는데 2에 가까우면 자기상관이 없다. 독립적이다.
#
# print('등분산성 ---')
# sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess = True, line_kws={'color':'red'})
# plt.show()  # 등분산성 만족
#
# print('다중 공선성 ---')
# vifdf = pd.DataFrame()
# vifdf['vif_value'] = [variance_inflation_factor(df.iloc[:, [0,2,3,5,6]].values, i) 
#                       for i in range(df.iloc[:, [0,2,3,5,6]].shape[1])]




print('단순선형회귀---')
model1 = smf.ols('mpg ~ hp',data=mtcars).fit()
print(model1.summary())
#  Prob (F-statistic):    1.79e-07  1.확인 <0.05 작으니 유의함
# R-squared:      0.602 2.확인 독립변수가 종속변수를 60.2% 설명한다. >15%
#Intercept     30.0989 절편 (회귀계수)
#hp            -0.0682 (기울기)
#P>|t| hp 비교 <0.05 이면 자격 OK
print(model1.summary().tables[0]) # 위에것만 보기

# y = -0.0682  * x + 30.0989

# print('다중선형회귀---') # 독립변수 2개이상
# model2 = smf.ols('mpg ~ hp+wt',data=mtcars).fit()
# print(model2.summary())
# # Prob (F-statistic):           9.11e-12 <0.05 OK
# # Adj. R-squared:                  0.815 81% >15% OK
# # Intercept     37.2273    P>|t|0.000 <0.05 OK
# # hp            -0.0318    P>|t|0.001 //    OK
# # wt            -3.8778    P>|t|0.000 //    OK
#
# # y = -0.0318  * x1(hp) + -3.8778 * x2(wt) + 37.2273
#
# # 예측 :
# new_pred = model2.predict({'hp':[110,150,90],'wt':[4,8,2]})
# print(new_pred)


# 선형회귀모델 자성 : LinearRegression 클래스 사용
from sklearn.linear_model  import LinearRegression
# lmodel = LinearRegression().fit(x,y) # 학습 후 모델을 생성

# 선형회귀모델의 과적합 방지용 클래스
# Linear Regression의 기본 알고리즘에 오버피팅 방지 목적의 제약조건을 담은 Ridge, Lasso, ElasticNet 회귀모형이 있다.

# 모델학습은 train dataset으로 모델검증은 test dataset 으로 하기
# train dataset, test dataset으로 나누기 

# 회귀분석 방법 1 - LinearRegression
# from sklearn.linear_model import LinearRegression
# print(train_set.iloc[:, [2]])  # petal length (cm), 독립변수
# print(train_set.iloc[:, [3]])  # petal width (cm), 종속변수
#
# # 학습은 train dataset 으로 작업
# model_linear = LinearRegression().fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])
# print('slope : ', model_linear.coef_)  # 0.42259168
# print('bias : ', model_linear.intercept_)  # -0.39917733
#
#
#
# # 모델 평가는 test dataset 으로 작업
# pred = model_linear.predict(test_set.iloc[:, [2]])
# print('예측값 : ', np.round(pred[:5].flatten(),1))
# print('실제값 : ', test_set.iloc[:, [3]][:5].values.flatten())
#
# from sklearn.metrics import r2_score
# print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred)))  # 0.93833( 결정계수) 93.8% 설명력을 가진다.
#
# print('\nRidge -----------')
# # 회귀분석 방법 - Ridge: alpha값을 조정(가중치 제곱합을 최소화)하여 과대/과소적합을 피한다. 다중공선성 문제 처리에 효과적. 
# from sklearn.linear_model import Ridge
# model_ridge = Ridge(alpha=10).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])
#
#
#
# #점수
# print(model_ridge.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]]))  # 0.91880
# print(model_ridge.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))    # 0.94101
# pred_ridge = model_ridge.predict(test_set.iloc[:, [2]])
# print('ridge predict : ', pred_ridge[:5])
#
#
#
# # plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='red')
# # plt.plot(test_set.iloc[:, [2]], model_ridge.predict(test_set.iloc[:, [2]]))
# # plt.show()
#
# print('\nLasso -----------')
# # 회귀분석 방법 - Lasso: alpha값을 조정(가중치 절대값의 합을 최소화)하여 과대/과소적합을 피한다.
# from sklearn.linear_model import Lasso
# model_lasso = Lasso(alpha=0.1).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])
#
#
#
# #점수
# print(model_lasso.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])) # 0.913863
# print(model_lasso.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))   # 0.940663
# pred_lasso = model_lasso.predict(test_set.iloc[:, [2]])
#
# print('lasso predict : ', pred_lasso[:5])
#
#
#
# # plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='blue')
# # plt.plot(test_set.iloc[:, [2]], model_lasso.predict(test_set.iloc[:, [2]]))
# # plt.show()
#
#
# # 회귀분석 방법 4 - Elastic Net 회귀모형 : Ridge + Lasso의 형태로 가중치 절대값의 합(L1)과 제곱합(L2)을 동시에 제약 조건으로 가지는 모형
# print('\nLasso -----------')
# # 회귀분석 방법 - ElasticNet: alpha값을 조정(가중치 절대값의 합을 최소화)하여 과대/과소적합을 피한다.
# from sklearn.linear_model import ElasticNet
# model_elastic = ElasticNet(alpha=0.1).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])
#
#
#
# #점수
# print(model_elastic.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])) # 0.913863
# print(model_elastic.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))   # 0.940663
# pred_elastic = model_elastic.predict(test_set.iloc[:, [2]])
# print('ElasticNet predict : ', pred_elastic[:5])
#
#
# # plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='cyan')
# # plt.plot(test_set.iloc[:, [2]], model_elastic.predict(test_set.iloc[:, [2]]))
# # plt.show()

# 비선형 회귀모델 : 선형 가정이 어긋날 때 (정규성을 만족하지 못할 때)
# 대처할 수  있는 방법으로 다항회귀모델 가능

# 비선형회귀 모델 작성(PolynomialFeatures)
# from sklearn.preprocessing import PolynomialFeatures # 다항식 특징을 추가가능
# poly = PolynomialFeatures(degree=2, include_bias = False) # degree 열 갯수 degree열의 갯수를 늘려서 할 수록 잔차가 줄어듬
# # 잔차가 너무 줄어들면 오버피팅 되서 너무 높게 설정 x
# x2 = poly.fit_transform(x) # 특징 행렬을 만들기
# print(x2)
#
# model2 = LinearRegression().fit(x2, y)
# ypred2 = model2.predict(x2)
# print(ypred2)
#
# plt.scatter(x, y)
# plt.plot(x, ypred2, c='red')
# plt.show()


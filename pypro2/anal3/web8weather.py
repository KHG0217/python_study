# 기상청 중기 예보 xml문서 읽기 
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


url ="https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"
data = req.urlopen(url).read()
# print(data.decode('utf-8'))

soup = BeautifulSoup(data, 'lxml')
# print(soup)

title = soup.find('title').string
print(title)

city =  soup.find_all('city')
# print(city) # [<city>서울</city>, <city>인천</city>

cityDatas =[]
for c in city:
    # print(c.string)
    cityDatas.append(c.string)

df = pd.DataFrame()
df['city'] = cityDatas
print(df.head(3))

tempMins = soup.select("location > province + city + data > tmn") 
#  >:직계 / 형제일때(+,-) +:아랫방향/ - :윗 방향/ city에 바로 다음 data

tempDatas =[]
for t in tempMins:
    tempDatas.append(t.string)
    
df['temp_min'] = tempDatas
print(df.head(3), len(df))
print()

df.columns = ['지역','최저기온']
print(df.head(3), len(df))

df.to_csv('날씨정보.csv', index = False)

print('~~~~~~~~~~~~~')
print(df.iloc[0], type(df.iloc[0]))
print()

print(df.iloc[0:2], type(df.iloc[0:2])) # df.iloc[0:2 :]
print()

print(df.iloc[0:2, 0:1], type(df.iloc[0:2]))
print()

print(df.iloc[0:2, 0:2], type(df.iloc[0:2]))
print()

print(df['지역'][0:2]) # 지역 칼럼 0행과 1행
print(df['지역'][:2])  # 지역 칼럼 0행과 1행
print()

print(df.loc[1:3]) # [1:3] = 인덱싱
print()

print(df.loc[[1,3]])# 1행과 3행
print()

print(df.loc[:, '지역'].head(2)) # 모든행, 지역열만
print()

print(df.loc[1:3, ['최저기온','지역']]) # 모든행, 최저기온,지역 열만
print()

print(df.loc[:, '지역'][1:4]) # 지역행 1~3까지만 
print()

print(df.loc[2:5, '지역'][0:3]) #2행에서 5행까지의 지역열중  0,1,2 행 출력
print()

print('-'*50)
print(df.info())
df = df.astype({'최저기온': 'int'}) # 문자를 숫자로 형변환
print(df.info())
print()

print(df['최저기온'].mean()) # 최저기온의 평균
print()

print(df.loc[df['최저기온'] >= 23]) # 23도 이상 칼럼 출력
print()

print(df.sort_values("최저기온", ascending=True)) # 온도 낮은온도부터 출력


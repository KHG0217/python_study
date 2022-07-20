import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

url ="https://kyochon.com/menu/chicken.asp"

data = req.urlopen(url).read().decode()
# print(data)



soup = BeautifulSoup(data, 'lxml')
# print(soup)

#tabCont01 > ul > li:nth-child(1) > a > dl > dt
name = soup.select("a > dl > dt")
#tabCont01 > ul > li:nth-child(1) > a > p.money > strong
price = soup.select("a > p.money > strong")
# print(name)
# print(price)


nameData=[]
for n in name:
    nameData.append(n.string)
    
# print(nameData)

priceData=[]
for p in price:
    priceData.append(p.string.replace(',', ''))
    
# print(priceData)

df = pd.DataFrame()
df['name'] = nameData 
df['price'] = priceData

print(df)
# df.info()
df = df.astype({'price':'int'})
print('가격평균:',round(df['price'].mean(),2)) # 평균
print('가격표준편차:',df['price'].std()) # 표준편차 




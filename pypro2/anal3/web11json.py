# 강남구 도서관 정보 json 문서 읽기
# 데이터 읽어오는곳 : 파일(scv,txt등),웹,데이터베이스
import json
import urllib.request as req
url = "https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json"

plainTxt = req.urlopen(url).read().decode()
print(plainTxt, type(plainTxt)) #  <class 'str'>
print()

jsonData = json.loads(plainTxt)
print(jsonData, type(jsonData)) # <class 'dict'>
print(jsonData['SeoulLibraryTime']['row'][0]['LBRRY_NAME'])

# get 함수
libData = jsonData.get('SeoulLibraryTime').get('row')
# print(libData)
print()

datas=[]
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    print(name+ ' ', tel)
    imsi = [name, tel]
    datas.append(imsi)
print()
    
import pandas as pd
df = pd.DataFrame(datas, columns=['도서관명','전화'])
print(df)
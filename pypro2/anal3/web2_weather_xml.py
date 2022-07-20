# 기창정 제공 날씨정보 XML문서 처리
import urllib.request
import xml.etree.ElementTree as etree

try:
    webdata = urllib.request.urlopen("http://www.kma.go.kr/XML/weather/sfc_web_map.xml")
    webxml = webdata.read()
    webxml = webxml.decode('utf-8')
    print(webxml)
    webdata.close()
    
    with open('weather_xml.xml', mode = 'w' , encoding = 'utf-8') as f:
        f.write(webxml)
except Exception as e:
    print('err : ', e)
 
xmlfile = etree.parse("weather_xml.xml")   

root = xmlfile.getroot() # root element를 얻을 수 있다
print(root.tag) # items
print(root[0].tag)

childeren = root.findall("{current}weather")
print(childeren)

for it in childeren:
    y = it.get('year') # 속성값 얻기
    m = it.get('month') # 속성값 얻기
    d = it.get('day') # 속성값 얻기
    h = it.get('hour') # 속성값 얻기
    print(y + "년 " + m + "월 " +d + "일 "+ h + "시 현재 ")
    
datas = []
for child in root:
    for it in child:
        # print(it.tag)
        local_name = it.text # 지역명 얻기
        #온도,상태 얻기
        re_ta = it.get("ta")
        re_desc = it.get("desc")
        # print(local_name,re_ta,re_desc)
        datas +=[[local_name,re_ta,re_desc]]
        
print(datas)

import pandas as pd
df = pd.DataFrame(datas, columns=['지역','온도','상태'])
print(df.head(3))
print(df.tail(3))
print()
print(len(df))

import numpy as np
imsi = np.array(df.온도, np.float32) # float로 형변환
# print(imsi)
print('평균온도: \n',round(np.mean(imsi), 2)) # 소수점 2번째 자리까지 출력
    
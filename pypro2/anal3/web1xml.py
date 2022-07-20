# 웹문서 처리 : XML ( 환경파일처리, DATABASE, WEB화면 디자인, 모바일화면 디자인)
# ElementTree 모듈 이용 

import xml.etree.ElementTree as etree

xmlfile = etree.parse("my.xml") # parse: ElementTree class 타입으로 바꿔주는것
print(xmlfile, type(xmlfile)) # ElementTree object

root = xmlfile.getroot() # root element를 얻을 수 있다
print(root.tag) # items
print(root[0].tag) # item / root의 0번째 element를 얻는다.
print(root[0][0].tag) # name / root 0번쨰의 0번째 element를 얻는다.
print(root[0][0].attrib) # root 0번쨰의 0번째 attribute를 얻는다 ({'id': 'ks1'})
print(root[0][2].attrib.keys()) # dict_keys(['kor', 'eng'])
print(root[0][2].attrib.values()) # dict_values(['100', '90'])
print()

imsi = list(root[0][2].attrib.values()) # 벨류값 리스트로 담기
print(imsi) 

print('--'*50)
children = root.findall('item') # root 자식에 있는 item을 모두 찾아줌

for it in children:
    re_id = it.find('name').get('id') # 속성값 id 얻기
    re_name = it.find('name').text # 요소값
    print(re_id,re_name)
    



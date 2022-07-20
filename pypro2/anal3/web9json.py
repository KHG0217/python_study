# JSON data <-> dict type의 호환
import json

dict = {'name':'tom', 'age':33, 'score':['90','80','100']} # python의 dict

print('dict:%s'%dict)
print('dict type:%s'%type(dict))

print('-----JSON encoding : dict, list, tuple등을 JSON 모양의 문자열로 변경-------')
str_val = json.dumps(dict) # dict -> str
print('dict:%s'%str_val)
print('dict type:%s'%type(str_val))
print(str_val[0:20]) # 문자열 관련 함수를 사용 할 수 있음을 알 수 있다.(문자열 관련 슬라이싱 가능)
# print(str_val['name']) <- dict에서 쓰는 명령어 오류

print('-----JSON decoding : JSON 모양의 문자열을 dict로 변경-------')
json_val = json.loads(str_val) # str -> dict
print('dict:%s'%json_val)
print('dict type:%s'%type(json_val))
# print(json_val[0:20]) # str에서 사용하는 명령어 오류
print(json_val['name']) # <- dict 명령어 오류 x
print()

for k in json_val.keys():
    print(k)
print()    

for k in json_val.values():
    print(k)


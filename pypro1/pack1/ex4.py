# 정규 표현식: 정규 표현식(영어: regular expression) 또는 정규식은 
#특정한 규칙을 가진 문자열의 집합을 표현하는 데 사용하는 형식 언어이다
import re

ss = "1234 56 abc가나다\mbcnabcABC_123556_6python is fun파이썬 만세"

print(re.findall('123', ss)) #ss에서 '123'만 찾아준다 list 타입
print(re.findall(r'가나',ss)) #ss에서 '가나'를 찾아준다,r을 붙여주면 이스케이프가 아니라고 알려준다.(특수문자 찾을때)
print(re.findall(r'[0-9]',ss)) #ss에서 0~9까지 한글자씩 찾아준다.
print(re.findall(r'[0-9]+',ss)) #*,+?,{횟수} +1개이상:1~9다음 있으면 다읽어줌
print(re.findall(r'[0-9]*',ss)) #0개이상
print(re.findall(r'[0-9]?',ss)) #?0개거나 1개
print(re.findall(r'[a-z,A-B]',ss))
print(re.findall(r'[a-z,A-B]+',ss))
print(re.findall(r'[^a-z,A-B]+',ss)) # ^부정 -쓴거 빼고
print(re.findall(r'[가-힣]+',ss)) #한글은 가-힣
print(re.findall(r'[^가-힣]+',ss)) #한글을 빼고
print(re.findall(r'[0-9]{2,3}',ss)) #첫 글자가 0-9까지의 숫자로 시작하는 2글자짜리 3글자짜리
pa = re.compile('[0-5]+') #패턴 만들기
print(re.findall(pa,ss))#패턴 활용 
imsi = re.findall(pa, ss)#변수에 담기
print(imsi[0])
print()
print(re.findall(r'.bc',ss)) #. <-아무글자나 +bc를 찾아준다.
print(re.findall(r'a..',ss)) #a로 시작하고 ..<-두글자 아무거나 총 세글자
print(re.findall(r'^1',ss)) #첫글자가 1로시작하는거 찾기
                            #^:대괄호에 붙으면 부정, 없이쓰면 첫시작 문자가 1
print(re.findall(r'만세$',ss)) #$:마지막 글자가 만세     
print()
print(re.findall(r'\d',ss)) #\d:숫자 찾기
print(re.findall(r'\d+',ss)) #숫자뒤에 붙이기                      
print(re.findall(r'\d{2}',ss)) #숫자 두글자 
print(re.findall(r'\D',ss)) # \d를 뺀 나머지  
print(re.findall(r'\s',ss)) #공백이나 tap
print(re.findall(r'\S',ss)) #공백이나 tap 제외
print(re.findall(r'\w+',ss)) #숫자나문자 이어서
print(re.findall(r'\W+',ss)) #숫자나문자 제외하고 이어서
print(re.findall(r'\\',ss)) #\만 찾기

print()

m = re.match(r'[0-9]+',ss) #그룹화를 시키는것
print(m.group())

print()
p = re.compile('the', re.IGNORECASE) #flag 사용,대소문자를 구분없음
print(p.findall('The dog the dog')) #대소문자를 구분없이 the를 찾아온다.
p = re.compile('the') #flag가 없으면 the만 찾아온다.
print(p.findall('The dog the dog'))

print()
ss= """My name is tom.
I am happy """ #주석문자열을 이용하여 str문을 완성할 수 있다.
print(ss)

p = re.compile('^.+',re.MULTILINE) #flag 사용,여러행을 한번에 넣을때 사용 list에 넣음
print(p.findall(ss))
imsi = p.findall(ss) #변수에 담아서 행별로 읽어올 수 있다.
print(imsi[0]) 
print(imsi[1])


                               


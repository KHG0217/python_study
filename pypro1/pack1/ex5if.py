#조건 판단문 if
var = 4

if var >= 3:
    # print('크구나') #블럭을 들여쓰기로 작성
    # print('참입니다.') #줄맞추기 중요
    pass #참일때 아무것도 실행하지 않겠다.
else:
    print('거짓이구만')
        
print()
money = 1000
age = 23

if money >= 500:
    item = 'apple'
    if age <= 30:
        msg = 'young'
    else:
        msg = 'old'
else:
    item = '바나나'
    if age >= 20:
        msg = ''
    else:
        msg = '어린이'

print(item, msg)

print()
# jum = int(input('점수 입력:')) #java Scanner 역활 ,input 입력값 문자, 형변환 함수 int('123')숫자가됨, str(123)문자가됨
# jum = jum + 1
# print(str(jum) + '점')
jum = 70 
res = ''
if jum >= 90:
    res = 'a'
elif jum >= 70:
    res = 'b'
else:
    res = 'c'
print('res : ',res)

if 90 <= jum <= 100:
    res = 'a'
elif 70 <= jum < 90:
    res = 'b'
else:
    res = 'c'
print('res : ', res,' ','res:' + str(res))

print()
names = ['김혁규', '전해리', '최팀장']
if '전해리' in names:  # not in
    print('친구야')
else:
    print('누구?')  
    
print()
a = 'kbs'
b = 9 if a == 'kbs' else 11 # b는 a=='kbs'라면 9를 갖고 아니면 11을 갖는다
print('b:', b)   
    
a = 11
b = 'mbc' if a == 9 else 'kbs'
print('b:', b)  

a = 3 #고정값 = 변수대문자(약속) A = 3 <- 고정값(상수)
if a > 5:
    re = a * 2
else:
    re = a + 2
print('re: ',re) 

re = a * 2 if a > 5 else a + 2
print(re)

a = 3
print((a + 2, a * 2)[a > 5]) #(a + 2, a * 2) 거짓일때,참일때 / [a > 5]=거짓0,참1

a = 3
if a < 5:
    print(0)
elif a < 10:
    print(1)
else:
    print(2)
            
# print(0 if a <5 else 1 if a < 10 else 2) 위의 코드와 같다 (잘 사용하진 않음)                                         

print('종료')

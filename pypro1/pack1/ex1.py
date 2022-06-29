'''
여러 줄 
주석
'''

"""
여러 줄
주석
"""

#한 줄 주석
print("표준 출력장치로 출력") #파이썬 2점대 버전은 함수형태가 아님
var1 = '안녕 파이썬'
print(var1) #파이썬은 앞에를 띄어쓰기하면 오류 (띄어쓰면 블럭이됨){}
var1 = 3; print(var1) #명령어를 한줄에 구분할때 ; 끝에는 안붙여도 됨
var1 = '변수 선언시 타입을 적지 않음. 참조 데이터에 의해 타입이 결정' 
print(var1)

print()
a = 10
b = 12.3
c = b #값을 기억하는게 아니고 객체의 주소를 기억하는 것이다
print(a, b, c) #정수는 int 실수는 float이다, 다 class임을 알 수있음

print(id(a), type(a)) #id는 객체의 주소를 확인할 수 있는 함수
print(id(b), type(b))
print(id(c), type(c))
print(a is b, a == b) #False False    is -> 주소 비교, == ->값 비교
print(b is c, b == c) #True True

aa = [100]
bb = [100]
print(aa == bb, aa is bb) #True False 값은 같지만 주소가 다르기때문에 
print(id(aa), id(bb)) # 주소가 다른걸 알 수있다.

print()
A = 1; a = 2;
print(A, ' ',a) #파이썬은 변수선언시 대소문자를 구분하고있다.

# for = 1 # 키워드를 변수로 사용할 수가 없다. Ctrl+/로 주석처리 할 수있다.

import keyword
print('키워드 목록: ', keyword.kwlist)

print('\n 숫자 진법')
print(10, oct(10), hex(10), bin(10)) #10진수,8진수,16진수,이진수
print(10, 0o12, 0xa, 0b1010) 

print('\n 자료형 확인')
print(3,type(3))
print(3.4,type(3.4))
print(3 + 4j,type(3 + 4j))
print(True,type(True))
print('3',type('3'))

                    #묶음형 자료형
print((3,),type((3,))) #tuple
print([3],type([3])) #list
print({3},type({3})) #set
print({'key':3},type({'key':3})) #dict -> json타입과 궁합이 잘맞는다.


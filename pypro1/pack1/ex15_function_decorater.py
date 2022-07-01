# 함수 장식자 (function decorator) : Meta 기능이 있다. (Meta = 어떤 기능을 가지고 있다)
# 함수 장식자는 또 다른 함수를 감싼 함수다.
# 장식자는 포장된 함수로 교체하여 함수를 반환한다.
# java - 어노테이트 = python -데코레이터

def make2(fn):
    return lambda:'안녕 '  + fn()

def make1(fn):
    return lambda:'반가워 '  + fn()

def hello():
    return "홍길동"

hi = make2(make1(hello))
# make2는 make1에서 hello 함수를 실행하고 리턴된 값을 make1리턴값에 넣고 주소를 반환 -> 
#그 주소를 make2에 넣고 주소를 반환 -> 
#그 주소를 hi변수에 담음
print(hi())

#위에 코드를 장식자를 넣어 사용하고있다. (여기선 리턴주소를 담는 함수지만 장식자는 어떤 기능을 사용하는구나 라고 알면 된다.)

@make2
@make1
def hello2():
    return "홍길자"

print(hello2())

print()
hi2 = hello2() # 실행 결과를 치환
print(hi2)
hi3 =hello2 # 함수 주소를 치환
print(hi3())

#파이썬 반복: 반복문, 재귀함수


# 일급함수 : 함수 안에 함수 선언, 인자로 함수 전달, 반환값으로 함수 사용

def func1(a, b):
    return a + b

func2 = func1 #똑같은 주소의 객체를 치환한것
print(func1(3, 4))
print(func2(3, 4))

def func3(func):    #    인자로 함수 전달
    def func4():    #    함수 안에 함수 선언
        print('나는 내부 함수라고 해')
    func4()
    return func #    반한값으로 함수 사용

mbc = func3(func1)
print(mbc(3, 4))

print('축약함수 (Lambda - 이름이 없는 한 줄짜리 함수)')#휘발성 함수
# 형식 : Lambda 인자, ...: 표현식

def Hap(x, y):
    return x + y

print(Hap(1, 2))
print((lambda x, y:x + y)(1,2)) #Lambda 형태

g = lambda x, y:x * y #lambda의 객체가 생성되고 g가 참조함
print(g)
print(g(3, 4))

print()
# lambda도 가변인수 사용이 가능하다.
kbs = lambda a, su=10: a + su
print(kbs(5))
print(kbs(5, 6)) #su가 6으로 오버라이드(덮어쓰기됨)
sbs = lambda a, *tu, **di:print(a, tu, di)
sbs(5, 7, 8, m=4, n=5)

print()
# List에 람다를 넣어 사용
li = [lambda a, b: a + b, lambda a, b: a * b]
print(li[0](3, 4))
print(li[1](3, 4))

print()
for i in li:
    print(i(3,4))
    
print('람다 적용해 보기: 다른 함수에서 람다를 속성으로 사용')
# filter(function, iterable) filter사용하기
print(list(filter(lambda a:a < 5, range(10)))) #5미만인 숫자만 list에 들어감
print(list(filter(lambda a:a % 2, range(10)))) #true인 홀수만 찍힘 2로 나눌때 0은 false 그외는 true
# 1 ~ 100 사이의 정수 중 5의 배수이거나 7의 배수만 출력
print(list(filter(lambda a:a % 5 == 0 or a % 7 == 0,range(1,101))))

*v1, v2, v3 = [1, 2, 3, 4, 5]
print(v1)
print(v2)
print(v3)

   

    



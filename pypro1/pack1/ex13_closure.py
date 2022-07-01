# closure(클로저) : scope에 제약을 받지 않는 변수들을 포함하고 있는 코드 블록이다.
# 함수 내에 선언된 변수를 함수 밖에서 참조가 가능핟록 하는 기술

def func_times(a, b):
    c = a * b
    # print(c)
    return c
    
print(func_times(2,3))

kbs = func_times(2, 3)
print(kbs)

kbs = func_times
# del func_times 함수 지우는법
print(kbs)
print(kbs(2, 3))
print(id(kbs), id(func_times)) #함수의 주소도 치환가능하다 kbs=func_time 주소값이 같다

mbc = sbs = kbs
print(mbc(2, 3))
print(sbs(2, 3))
print(kbs(2, 3))
del kbs
print(mbc(2, 3)) #kbs를 지웠지만 mbc와 sbs는 살아있음

print('\n클로저를 사용하지 않은 경우 -------------')
def out():
    count = 0
    def inn():
        nonlocal count
        count +=1
        return count
    print(inn())

# print(count)
out()
out()
out()

print('\n클로저를 사용한 경우 -------------')
def outer():
    count = 0
    def inner():
        nonlocal count
        count +=1
        return count
    return inner # 이것이 클로져: 내부함수의 주소를 반환
#                            함수 바깥에서 ->내부함수를 실행할 수 있다.
var1 = outer() #inner의 주소 #Car car1 =new car 느낌
print(var1)
print(var1())
print(var1())
print(var1())

print()
var2 = outer() #Car car2 =new car 느낌/ 새로운객체
print(var2())
print(var2())

print(id(var1), id(var2)) #주소는 다르지만 inner 내부함수를 쓰고있다
    
print('클로저 써먹기')
# 수량 * 단가 * 세금 출력하는 함수 작성
def outer2(tax):    #tax 지역변수    outer2에서만 유효
    def inner2(su, dan):
        amount = su* dan * tax
        return amount
    return inner2 #내부함수를 반환하는 클로저

#1분기에는 금액에 대한 세금(tax)이 0.1이 부과
q1 = outer2(0.1)
result1 =q1(5, 50000)
print('result1 : ',result1)

print()
result2 =q1(2, 10000)
print('result2 : ',result2)

#2분기에는 금액에 대한 세금(tax)이 0.05이 부과
q2 = outer2(0.05)
result3 =q2(5, 50000)
print('result3 : ',result3)




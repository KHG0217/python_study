# 함수 (사용자 정의)
a = 10
b = a
print(a)
print('뭔가를 하다가 함수가 필요하면 선언' )

# 함수 작성
def do_func1():
    print('do_func1 처리')
    
def do_func2(arg1, arg2):   # do_func2(para1, para2) parameter:매개변수
    tmp = arg1 + arg2
    # return tmp    #함수는 반환값이 꼭 있다, 리턴을 작성하지면 None을 반환한다.
    """    
    if tmp % 2 == 1:
        return      #리턴뒤에는 함수 종료 리턴뒤에 코딩해도 읽지x(죽은문장)
                    #홀수이면 None을 반환
    else:
        return tmp #
    """
    if tmp % 2 == 1:
        return #홀수면 none
    print("tmp출력 : ", tmp) #if문이 아니면 찍고 none을 반환
    
# 함수 호출
aa = do_func1 #주소를 치환한것
print(do_func1) 
print(aa)
do_func1()
print('b : ', b)
do_func1() #자기를 불렀던 곳으로 돌아가서 코드실행하고 다시 돌아옴 (함수 싸이클)
print('-------------')
bb = do_func2(5, 5) #argument:전달인자, 인자
print('반환된 bb는', bb)   

print('--' * 20)
def area_tri(a, b):
    print('a')
    result = a * b /2
    area_print(result) #함수가 다른함수를 호출할 수 있다.
    print('c')

def area_print(result):
    print('삼각형의 면적은: ',str(result))
    print('b')
    
area_tri(5,4)
print()

def swap_func(a, b):
    return b, a #두개의값이 넘어간게 아니고 집합형 자료 하나로 넘어간다 
                #(tuple type으로 묶여 하나의 값으로 반환)

a = 10; b = 20
print(a, b)
imsi = swap_func(a, b)
print(imsi)

a,b = swap_func(a, b) 
print(a, b)

print()
def func1():
    print('func1 처리')
    def func2():        #내부 함수 inner function
        print('func2 처리')
    func2() #func1의 소유물이기 때문에 func1에서 호출해야한다
    
func1()

#if 조건식에서 함수 사용
def isOdd(para):
    return para % 2 == 1 #true나 false를 반환하는 함수다

mydict = {x:x*x for x in range(11) if isOdd(x)}
print(mydict)

print('종료')







    

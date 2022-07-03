# 함수 : 인수(argument)와 매개변수(parameter)의 매칭
# 매개변수 유형: 위치 매개변수, 기본값 매개변수, 키워드 매개변수, 가변 매개변수:a,b,c(값이 바뀌는 변수)

#                    end=5 <- 기본값 매개변수
def show_gugu(start, end=5):
    for dan in range(start, end +1 ): #5까지 돌게하려고 +1
        print(str(dan) + '단 출력')
        
print('뭔가를 하다가...')
show_gugu(2, 3) #end에 3이 덮어씌워짐
print()
show_gugu(3) #end에 기본값 매개변수인 5가 자동으로 들어감
print()
show_gugu(start = 2, end = 3)  #키워드 매게변수 start =, end =
print()
show_gugu(end = 3, start = 2)
print() 
show_gugu(2, end = 3) #맨처음 키워드값은 생략가능하나 그뒤에 키워드값은 생략x
print()
#show_gugu(end=3, 2) #SyntaxError: positional argument follows keyword argument
#                    위치 매개변수 오류
print()
# show_gugu(start = 2,3) #위치 매개변수 오류/ 앞에 키워드를 써주면 뒤에 키워드도 써줘야함
print()

print('가변 인수 처리 ---')
def func1(*ar): #tuple(a, )o (a)x
    print(ar)
    for i in ar:
        print('음식:'+i)
    
func1('비빕밥', '공기밥', '볶음밥', '주먹밥')
print()

def func2(a, *ar):
# def func2(*ar, a): # missing 1 required keyword-only argument: 'a'
    print(a)
    for i in ar:
        print('배고프면:'+i)
    
func2('비빕밥', '공기밥', '볶음밥', '주먹밥')

print()
def select_process(choice, *ar): #키워드 매개변수
    if choice == 'sum':
        re = 0
        for i in ar:
            re += i
    elif choice == 'mul':
        re = 1
        for i in ar:
            re *= i
    
    return re
print(select_process('sum', 1,2,3,4,5))
print(select_process('mul', 1,2,3,4,5))

print('parameter가 dict type---')
def func3(w, h, **other):   #db연동프로그램할때 쓸것 **dict타입만
    print('몸무게{}, 키{}'.format(w, h))
    print(other)
    
func3(65, 176, name='신기해', old='23')
func3(65, 176, name='신선해')
# func3(65, 176, {'name':'신선해'}) # err

print()
def func4(a,b,*v1,**v2):
    print(a,b)
    print(v1)
    print(v2)
    
func4(1, 2)
print()

func4(1, 2, 3, 4, 5)
print()

func4(1, 2, 3, 4, 5, m=6, n=7)

    


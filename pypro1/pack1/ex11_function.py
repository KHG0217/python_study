# 변수의 생존 범위 (scope rule)
# 접근순서 Local > Enclosing function(내부함수) > Global
#        1        2                            3

player = '전국대표' # 전역변수 (모듈:module의 어디서든 호출 가능)

def func_soccer():
    name = '미스터 손' # 지역변수 (현재 함수 내에서만 유효)
    player = '지역대표'
    print(name, player)
    # print(f"{name} {player}") f String
print(func_soccer)
func_soccer()
# print(name) #오류: func_soccer 함수안에서만 유효
#                   NameError: name 'name' is not defined
              
print()
a = 10; b = 20; c= 30 # 전역변수
print('1) a:{}, b:{}, c:{}'.format(a,b,c))
def func():
    a = 40
    b = 50
    c = 7
    def inner_fnc():
        # c = 60 
        global c #(지역변수 -> 전역변수)
        nonlocal b #부모 함수의 변수수준
        print('2) a:{}, b:{}, c:{}'.format(a,b,c))
            
        c = 60
        #UnboundLocalError: local variable 'c' referenced before assignment
        #c가 출력되기전에 값을 갖게 만들어 주려면 수준을 바꿔준다      
        b = 70
    inner_fnc()
    print('3) a:{}, b:{}, c:{}'.format(a,b,c))
    
func()
print('4) a:{}, b:{}, c:{}'.format(a,b,c))
# 클래스의 이해

kor = 100 # 모듈의 멤버인 전역변수

def abc(): # 모듈의 멤버인 함수
    kor = 50 # 함수에서 유효한 지역변수
    print('모듈의 멤버인 함수')
    
class My: #모듈의 멤버인 클래스
    # 생성자는 생략 가능하다
    kor = 10 # class의 멤버인 지역변수
    def abc(self):
        print('클래스의 멤버인 메소드')
        
    def showData(self):
        kor = 30 # 메소드에서 유효한 지역변수
        print('kor : ', kor) # 메소드에서 유효한 지역변수인 kor/ 메소드에서 유효한 지역변수 kor = 30이 없다면, 전역변수인 k=100으로 감
        print('kor : ', self.kor) #class 내의 멤버 kor/ My.kor이라고 해도됨
        print()
        self.abc() # 클래스의 멤버인 메소드
        abc() # 모듈의 멤버인 함수
        
m = My() # m 은 My타입의 객체로 생성되었다
m.showData()

print('---------------')
class Our:
    a = 1
    
print(Our.a) # 1

our1 = Our()
print(our1.a) # 1
our1.a = 2
print(our1.a) # 2

our2 = Our()
print(our2.a) # 1

print(Our.a) # 1







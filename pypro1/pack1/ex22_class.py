# 모듈의 멤버 중 클래스를 이용해 객체지향(중심)적인 프로그래밍이 가능(다용성 구성등)
# 클래스는 새로운 이름공간을 지원하는 단위로 맴버메소드와 멤버변수를 갖는다.
# 접근지정자가 없다. (무조건 전역[public]) 메소드 오버로딩 없다.(클래스내에 똑같은 이름의 메소드 줄 수 없다.)
# __변수명__ : __ <- 시스템에서 자동 호출하게 하는 callback
# class는 새로운 type을 만드는 것이다. - 공통으로 가져야할 요소들을 작성할때 , 자원의 재활용
# class 의 자원의 재활용 방법2가지 - has a(class 의 포함관계), is a(class의 상속)
# class 포함 = class a, class b 그림에서 다이아모양
#    class ():
#        b()
# 상속은 잘 사용하지 x -> 유지보수가 어려움 (끈끈하게 이어져 있기 때문) 그림에서 세모(화살표모양)
a = 10
print('모듈의 멤버 중 a는', a, type(a))

# class 선언하기
class TestClass: # class 는 새로운 타입을 만든다 TestClass 타입이 생성되는 것 class 헤더
    #class 바디
    aa = 1 # 멤버 변수(전역)
    
    def __init__(self):   #self = 자바에 this = 여기선 TestClass
        print('생성자')    # 클레스의 객체가 생성되면서 제일먼저 실행되는 함수
        
    def printMsg(self): # class안에 있으면 class 멤버 메소드 (전역) / class밖에 있으면 모듈의 멤버로 함수
        name = '홍길동'    # 지역 변수
        print(name)
        print(self.aa)
        
    def __del__(self):
        print('소멸자')    # 클레스가 종료되면서 마무리로 실행
#                          파이썬에서 가비지컬렉터가 있어서 잘 사용하지 않는다.

print('원형 클래스의 주소 : ',id(TestClass)) # 원형 class는 프로그램 실행시 자동으로 객체화된다. 원형 class의 주소(설계도의 주소)
print(TestClass.aa)
# TestClass.printMsg(self) 클래스 이름으로 메소드를 부를수 없음 아규먼트를 줄 수 없음. 오류 


print()

test = TestClass() # 생성자를 호출함 TestClass type의 객체가 생성됨
print('TestCalss type의 새로운 객체의 주소 id(test) :', id(test))
print(test.aa)
test.printMsg() # Bound method call: 자동으로 객체변수가 인수로 전달(여기선 test의 주소가 자동으로 self로 들어감) 
print('---')
TestClass.printMsg(test) # unBound method call 
#클레스의 이름을 넘길때는 객체변수가 필요해서 test를 넘겨주는 것

print()
print(type(test)) #testClass 타입의 객체가 생성된 것 <class '__main__.TestClass'>
print(isinstance(test, TestClass))
del test # 객체는 만들어 졌지만 test가 사라진것 (가르키는 주소가 사라졌기때문에 객체는 가비지컬렉터에서 사라짐)
# print(isinstance(test, TestClass)) NameError: name 'test' is not defined
print('종료')


print('종료') # 종료가 찍히고 소멸자가 찍힘 => 정확히는 응용프로그램이 종료시점에 실행됨
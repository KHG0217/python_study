# 추상 클래스 - 추상 메소드 
# 하위 클래스에서 부모의 메소드를 반드시 오버라이드 하도록 강요가 목적
# 다형성을 구현하기 위함 (강요) 다형성(polymorphism)이란 하나의 객체가 여러 가지 타입을 가질 수 있는 것을 의미합니다

from abc import *

class AbstaractClass(metaclass = ABCMeta):  #추상 클래스 -> 객체로 만들어질 수 없다. -> 하지만 생성자는 만들 수 있다.
    
    @abstractmethod
    def aaMethod(self): # 추상 메소드
        pass
    
    def normalMethod(self): # 일반 메소드
        print('추상 클레스 내의 일반 메소드')
        
# p = AbstaractClass() #     p = AbstaractClass() TypeError: Can't instantiate abstract class AbstaractClass with abstract method aaMethod
# 추상 클래스는 객체로 만들 수 없다.

class Child1(AbstaractClass):   #추상 클래스를 부모로 부터 받으면 이 클래스도 추상 클래스가 된다.
    name = '난 Child1'
    
    def aaMethod(self): # 추상 메소드를 재정의 해주지 않으면 객체가 생성되지 않고 오류(오버라이딩을 강요 당함)
        print('추상메소드를 일반 메소드로 재정의')
    
c1 = Child1()
print(c1.name)
c1.aaMethod()
c1.normalMethod()

print('---------------')

class Child2(AbstaractClass):
    def aaMethod(self):
      print('Child2에서 추상메소드를 일반 메소드로 오버라이딩')
      a = 120
      print('a :', a - 20)  
    
    def normalMethod(self): # 일반 메소드
        print('Child2에서 부모와 기능이 다른 일반 메소드')
        
c2 = Child2()

print('다형성: 하나의 변수가 다향한 기능을 갖게 되는것')
kbs = c1
kbs.aaMethod()
kbs.normalMethod()
print()

kbs = c2
kbs.aaMethod()
kbs.normalMethod()

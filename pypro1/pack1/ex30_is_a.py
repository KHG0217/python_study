# 클래스의 상속 이해
# Person <- Employee
#        <- Worker <- Programmer

class Person:
    say = '난 사람이야 ~'
    nai = '20'
    __good = '체력을 다지자' #private 멤버가 됨 
    
    def __init__(self, nai):
        print('Person 생성자 호출')
        self.nai = nai  #새로운 객체에 nai가 만들어지는것
                        #self.nai가 없다면 Person.nai를 참조함
        
    def printInfo(self):
        print('나이:{}, 이야기:{}'.format(self.nai, self.say)) #self.nai -> __init__ => self.nai = nai
        
    def hello(self):
        print('안녕')
        print('private 멤버 출력 :', self.__good)
    
    @staticmethod     #Person class 안에 작성하였지만 static 함수라 모든 곳에서 참조가능
    def sbs(tel):
        print('staticmethod: 클래스 소속 - 클래스 멤버와 상관없는 독립적 처리를 할 경우에 사용')
        
print(Person.say, Person.nai)
p = Person('22')
p.printInfo()
p.hello()

print('---------Employee-----------')
class Employee(Person):
    say  = '일하는 동물'
    subject = '근로자'
    def __init__(self): # 없으면 오류
                        #이유: 없으면 부모의 생성자를 호출, 여기서는 def __init__(self, nai)
                        #    오류없이 실행하려면 아래코드 e = Employee()에 e = Employee(nai값)을 넣어줘야함
        print('Employee 생성자')
        
    def printInfo(self):
        print('Employee 클래스의 printInfo') 
        
    def e_printInfo(self):
        self.printInfo()    # 찾다가 없으면 부모 클래스로 올라감
        super().printInfo() # 바로 부모클래스로 올라감
    
e = Employee()
print(e.say, e.nai, e.subject)
# e.printInfo()
e.e_printInfo()
print('---------Worker-----------')
class Worker(Person):
    def __init__(self, nai):
        # self.nai = nai # Worker nai에 들어감
        print('Worker 생성자')
        super().__init__(nai) # Bound method call/ 
        # 부모의 생성자(Person)를 호출  -> 
        # self에 Worker주소를 담고,nai값을 들고 Person __init__ self에 가져감 -> 
        # Person 생성자에서 self.nai = nai 가 호출되어 Worker객체에 nai를 넣음 (Worker 에 nai가 생김/ Worker.nai = nai값) self = Worker 
 
        
    def w_printInfo(self):
        super().printInfo()
w = Worker('30')
print(w.say, w.nai) #nai가 없기때문에 공유주소
w.w_printInfo()
w.printInfo()

print('---------Programmer-----------')
class Programmer(Worker):
    def __init__(self,nai):
        print('Programmer 생성자')
        # super().__init__(nai) #Worker 에게 끌고 올라감, Bound method call
        Worker.__init__(self, nai) # unBound method call
        # 부모 생성자(Worker)의 생성자를 호출 ->
        # self에 Programmer 주소, nai에 값을 받아 ->Worker 생성자로 가져감
        # Worker 생성자에선 부모 생성자인(Person)에게 Programmer로부터 받아온 Programmer주소와, nai값을 보냄
        # Person 생성자에서  self.nai = nai 가 호출되어 값이 들어가고 (Programmer.nai = nai값) self = Programmer 
        # Programmer에 nai라는 객체가 생겨 참조한다.

    def w_printInfo(self):
        print('Programmer에서 부모 생성자 override')

    # def hello(self):
    #     print('private 멤버 출력 :', self.__good)
                
pr = Programmer('33')
print(pr.say, pr.nai)
pr.w_printInfo()
pr.printInfo()
print(w.say, w.nai)
# pr.hello() #AttributeError: 'Programmer' object has no attribute '_Programmer__good'
#             공유멤버였다면 Programmer을 참조하여 없으면 위에서 찾았겠지만 __good은 private 요소이기 때문에 class내에서만 실행이됨          
           

print('~~~~~~~~~~~~~~~')
print(type(1.2))
print(type(pr))
print(type(w))
print(Programmer.__base__) # __base__ 부모 클래스를 확인하는 함수
print(Worker.__base__)
print(Person.__base__) # Person의 부모는 object
pr.sbs('111-1111')
Person.sbs('222-22222') #권장


          
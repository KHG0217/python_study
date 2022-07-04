# 클래스의 상속 이해
# Person <- Employee
#        <- Worker <- Programmer

class Person:
    say = '난 사람이야 ~'
    nai = '20' 
    
    def __init__(self, nai):
        print('Person 생성자 호출')
        self.nai = nai  #새로운 객체에 nai가 만들어지는것
                        #self.nai가 없다면 Person.nai를 참조함
        
    def printInfo(self):
        print('나이:{}, 이야기:{}'.format(self.nai, self.say)) #self.nai -> __init__ => self.nai = nai
        
    def hello(self):
        print('안녕')
        
print(Person.say, Person.nai)
p = Person('22')
p.printInfo()
p.hello()

print('---------Employee-----------')
class Employee(Person):
    say  = '일하는 동물'
    subject = '근로자'
    def __init__(self): #? 없으면 오류
                        #이유: 없으면 부모의 생성자를 호출, 여기서는 def __init__(self, nai)
                        #    오류없이 실행하려면 아래코드 e = Employee()에 e = Employee(nai값)을 넣어줘야함
        print('Employee 생성자')
        
    def printInfo(self):
        print('Employee 클래스의 printInfo') #부모와 자식 메소드 모두 실행
        
    def e_printInfo(self):
        self.printInfo()    # 찾다가 없으면 부모 클래스로 올라감
        super().printInfo() # 바로 부모클래스로 올라감
    
e = Employee()
print(e.say, e.nai, e.subject)
# e.printInfo()
e.e_printInfo()  
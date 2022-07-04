# 자원의 재활용을 목적으로 클래스를 상속 가능.
# 이때 다중 상속도 허용하고 있다.

class Animal:
    nai = 1
    
    def __init__(self):
        print('Animal 생성자')
    
    def move(self):
        print('움직이는 생물')
        
class Dog(Animal):  #Dog은 Animal의 자식이된다 파이썬에서 상속시키는법 class 이름(부모class)
    # dog = Animal() #클래스 포함
    irum = '난 댕댕이'
    
    def __init__(self): #부모 자식의 모두 생성자가 있을때 자식의 생성자만 수행, 자식 생성자가 없다면 부모 생성자 호출
        print('Dog 생성자')
    
    def my(self):
        print(self.irum + ' 만세')
    pass
dog1 = Dog()
dog1.my()
dog1.move() #dog을 뭔저 뒤지고 없으면 -> 부모로 올라감
print('nai :', dog1.nai)

print()
class Horse(Animal):
    pass
Horse1 =Horse()
Horse1.move()
print('nai :', Horse1.nai)
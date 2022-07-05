# 다중상속 연습: 카페 연습문제 3번 class는 모듈별로 따로 만들어야함(원래는)

class Animal:
    def move(self):
        pass
    
class Dog(Animal):  # 단일 상속
    name = '개'
    
    def move(self):
        print('댕댕이는 낮에 활기있게 돌아다님')

class Cat(Animal):
    name = '고양이'
    
    def move(self):
        print('애옹s는 밤에 돌아다님')
        print('눈빛이 빛남')
        
class Wolf(Dog, Cat):   # 다중 상속
    pass

class Fox(Cat, Dog):
    def move(self):
        print('여우처럼 민첩하게 돌아다님')
        
    def FoxMethod(self):
        print('여우 고유 메소드')

dog = Dog()
print(dog.name)
dog.move()

print()

cat = Cat()
print(cat.name)
cat.move()

print()

wolf = Wolf()
fox = Fox()

ani = wolf
ani.move() #부모 Dog의 move()

print()

ani = fox
ani.move()

print('^^^^' * 10)
anis = [dog, cat , wolf, fox]
for a in anis:
    a.move()
    
print()
print(Fox.__mro__) # 다중상속일때 읽는 순서가 나와있음 왼쪽 -> 오른쪽 순 !<class '__main__.Cat'>, <class '__main__.Dog'>
print(Wolf.__mro__)# <class '__main__.Dog'>, <class '__main__.Cat'>




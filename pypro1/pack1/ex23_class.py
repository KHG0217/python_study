# 클래스의 이해
print('어쩌구 저쩌구 하다가 ...')

class Car:
    handle = 0
    speed = 0
    test = 'car1에서 원형클레스인 car의 변수'
    
    def __init__(self, speed, name): #이번엔 객체변수를 만들때 speed와 name을 줘야함
        self.speed = speed
        self.name = name
        
    def showData(self):
        km = '킬로미터'
        msg = '속도: ' + str(self.speed) + km
        return msg + ', 핸들은' + str(self.handle)
    
print(Car.handle, Car.speed)
print()

car1 = Car(5, 'tom') #car 1의 주소가 객체변수인 self로 자동으로 들어감
print(car1.handle, car1.speed, car1.name, car1.test) #car1에 찾는게 없다(handle) 원형클레스인 car를 참조함
car1.color = '파랑' #car1 객체에 color 라는 변수에 '파랑'을 만들어 추가함
print('car1 자동차 색은', car1.color)
print()     

car2 = Car(10, 'jone')
print(car2.handle, car2.speed, car2.name)
# print('car2 자동차 색은', car2.color) #AttributeError: 'Car' object has no attribute 'color' 오류

print('method')
print('car1:',car1.showData()) # self 는 car1
print('car2:',car2.showData()) # self 는 car2
print()
car1.speed = 100
Car.handle = 1 #원형클레스의 지역변수를 1로 바꿈
print('car1:',car1.showData()) # self 는 car1
print('car2:',car2.showData()) # self 는 car2
print('원형클래스의 speed:', Car.speed)

print()
print(id(Car), id(car1), id(car2)) #모두 주소가 다르다는 것을 알 수있음
print(type(car1), type(car2)) # 모두 타입은 같다는 것을 알 수있음
print()
print(car1.__dict__) #객체의 멤버를 확인(__dtct__)
print(car2.__dict__)



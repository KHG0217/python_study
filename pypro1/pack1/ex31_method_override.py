# method override(재정의)

class Parent:
    def printData(self):    #pass로 비어둔경우는 override해서 사용해라 라는 뜻으로 생각
        pass

class Child1(Parent):
    def printData(self):
        print('Child1에서 override')
        
class Child2(Parent):
    def printData(self):
        print('Child2에서 override')
        print('부모 메소드와 이름은 같으나, 다른 기능을 가짐')
        
    def abc(self):
        print('Child2 고유 메소드')        
c1 = Child1()
c1.printData()
print()
c2 = Child2()
c2.printData()

print('다형성 처리')
# par = Parent() # 부모객체에게 자식의 주소를 줄 수 있다 /하지만 파이썬은 상관 x = 주소를 가르키는거기때문에 상관이없다 
par = c1 #자식 class로 치환
par.printData
print()
par = c2
par.printData
print()
plist = [c1, c2]
for i in plist:
    i.printData()
    print()

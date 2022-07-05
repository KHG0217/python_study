# 다중 상속

class Tiger:
    data = '호랑이 세계'
    
    def cry(self):
        print('호랑이 : 어흥 ;; 휴 더워;;;;')
    
    def eat(self):
        print('맹수는 고기를 좋아함. 고기 먹은 후 아아를 마심')

class Lion:
    
    def cry(self):
        print('사자 : 으르렁 ;; 겁나 더우셈;;;;')
        
    def hobby(self):
        print('백수의 왕은 낮잠을 즐김')
               
    
class Liger1(Tiger, Lion):  #다중상속은 순서가 중요하다!
    pass

l1 = Liger1()
l1.cry() #동일한 멤버(여기서는 동일한 메소드) 가 있을때 먼저 상속한게 상속 됨 Tiger
l1.eat() # Tiger
l1.hobby() # Lion
print(l1.data) #Tiger

def hobby():
    print('이건 함수라고 해')

print('------------------')
class Liger2(Lion, Tiger):
    data = '라이거 만세'
    
    def play(self):
        print('라이거 고유 메소드')
        
    def hobby(self):
        print('라이거는 프로그램 짜기가 취미')
        
    def showData(self):
        self.hobby() #지역함수 Liger2
        super().hobby() #부모함수 Lion
        hobby() #전역함수
        self.eat() # 지역함수에 x -> Lion에 x ->그다음 상속인 Tiger
        super().eat()# Lion에 x -> Tiger
        print(self.data +' '+ super().data)
l2 = Liger2()
l2.play()
l2.showData()

        
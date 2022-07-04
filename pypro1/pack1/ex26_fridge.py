# 냉장고 클래스에 음식 클래스 담기 - class 포함 연습

class Fridge:
    isOpend = False
    foods = [] #list에 담기
    def open(self):
        self.isOpend = True
        print('냉장고 문 열기')
        
    def put(self, thing):
        if self.isOpend:
            self.foods.append(thing) #클래스의 포함
            print('냉장고 속에 음식을 저장함')
            self.list()
        else:
            print('냉장고문이 닫혀서 음식을 저장할 수 없어요')
            
    def list(self): # 냉장고 속 음식물 목록 확인
        for a in self.foods:
            print('-',a.irum, a.expiry_date)
        print()
        
    def close(self):
        self.isOpend = False
        print('냉장고 문 닫기')
        
        
class Food_data:
    def __init__(self, irum, expiry_date):
        self.irum = irum
        self.expiry_date =expiry_date
        
f =Fridge()

apple = Food_data('사과', '2022-07-10')
f.put(apple)
f.open() 
f.put(apple)
f.close()

print()
tera = Food_data('테라', '2023-12-10')
f.open()
f.put(tera)
f.close()
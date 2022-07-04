# 커피 자판기 프로그램

class CoinIn:   #계산
    # coin =  0# 넣은 동전
    # change = 0 #거스름 돈
   
    def culc(self, cupCount):
        result = ''      
        if self.coin < 200:
            result = '요금부족'
        elif cupCount * 200 > self.coin:
            result = '요금부족'
        else:    
            self.change = self.coin - (cupCount * 200)
            result ='커피 '+ str(cupCount) + '잔과 잔돈' +str(self.change)+ '원'
        return result     
                
        

class Machine:
    # cupCount=1
    def __init__(self):
        self.coInIn = CoinIn() # 생성자에서 일어난 클래스의 포함
    
    def showData(self):
        self.coinIn.coin = int(input('동전을 입력하세요'))
        self.cupCount = int(input('몇 잔 을 원하세요'))
        print(self.coinIn.calc(self.cupCount))
        
Machine().showData()        
       
    
from abc import *

class Employee(metaclass = ABCMeta):

    def __init__(self,irum,nai):
        self.irum = irum
        self.nai = nai
        
        print('Employee 생성자 호출')
    
    @abstractmethod
    def pay(self):
        pass
    
    @abstractclassmethod
    def data_print(self):
        pass
    
    def irumnai_print(self):
        print('이름:{},나이:{}'.format(self.irum, self.nai))
    
class Temporary(Employee):
    
    def __init__(self,irum,nai,ilsu,ildang):
        super().__init__(irum, nai)
        self.ilsu = ilsu
        self.ildang = ildang
        
    def pay(self):
        self.result = self.ilsu * self.ildang
        
    def data_print(self):
        self.pay()
        print('이름:{},나이:{},월급:{}'.format(self.nai,self.irum,self.result))
        
t = Temporary("홍길동",25,20,15000)
t.data_print()


class Regular(Employee):
    
    def __init__(self,irum,nai,salary):
        super().__init__(irum, nai)
        self.salary = salary
    
    def pay(self):
        pass    
        
    def data_print(self):
        print('이름:{},나이:{},급여:{}'.format(self.irum,self.nai,self.salary))
            
r = Regular("한국인",27,3500000)
r.data_print()

class Salesman(Regular):

    
    
    def __init__(self, irum,nai,salary,sales,commission):
        Regular.__init__(self, irum, nai, salary)
        self.sales = sales
        self.commission = commission
        
    def pay(self):
        self.result=self.salary + (self.sales * self.commission) 
        
    
    def data_print(self):
        self.pay()
        print('이름:{},나이:{},수령액:{}'.format(self.irum,self.nai,self.result))
        
        
s = Salesman("손오공",29,1200000,5000000,0.25)
s.data_print()            
    
    
    
        
        
    
           
    
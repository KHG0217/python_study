# file i/o

import os

print(os.getcwd()) #현재 모듈의 경로

try:
    print('파일 읽기')
    # f1 = open(r'C:\Users\acorn\Desktop\GitRepository\python_study\pypro1\pack2\ftest.txt', mode='r' , encoding='utf-8') #이스케이프 문자 안읽기 위해 r
    # f1 = open(os.getcwd() + r'\ftest.txt', mode='r' , encoding='utf-8')
    f1 = open('ftest.txt', mode='r' , encoding='utf-8') # 현재경로일때는 경로를 생략해도 된다.
    print(f1.read())
    f1.close() # 메모리 효율적인 관리를 위해 close로 닫아주기
    
    print('파일 저장')
    f2 = open('ftest2.txt', mode='w', encoding='utf-8')
    f2.write('손오공\n')
    f2.write('사오정\n')
    f2.write('저팔계\n')
    f2.close()
    
    print('파일 추가')
    f2 = open('ftest2.txt', mode='a', encoding='utf-8')
    f2.write('김치국\n')
    f2.write('공기밥\n')
    f2.close()    

except Exception as e:
    print('err : ', e)
    
print('with 구문을 사용하면 close() 자동 처리 ----')
try:
    #저장
    with open('ftest4.txt', mode='w', encoding='utf=8') as obj1: # with 을 쓰면 close()를 안해도 자동으로 닫아준다. # mode = 생략가능
        obj1.write('파이썬으로 파일 처리\n')
        obj1.write('with 처리\n')
        obj1.write('close 생략\n')
        
    # 읽기
    with open('ftest3.txt', 'r', encoding='utf=8') as obj2: 
        print(obj2.read())

        
except Exception as e2:
    print('err2 : ', e2)

print()    
print('피클일(객체를 파일로 저장 및 읽기) ----')
import pickle

try:
    #개체 저장
    dicData = {'tom':'111-1111', 'john':'222-2222'}
    listData = ['장마철', '장대비 예고']
    tupleData = (dicData, listData)
    
    with open('hello.data', mode='wb') as ob1:  #hello.data파일을 만들고 tupleData,listData의 내용을 넣음
        pickle.dump(tupleData, ob1)   # pickle.dump(대상, 파일객체)
        pickle.dump(listData, ob1)
        
    #객체 읽기
    with open('hello.data', mode='rb') as ob2:
        a, b = pickle.load(ob2) # 먼저저장한 순서대로 a,b에 들어감
        print(a)
        print(b)
        print()
        c =  pickle.load(ob2)
        print(c)
            
except Exception as e3:
    print('err3 : ', e3)    
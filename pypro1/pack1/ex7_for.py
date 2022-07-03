# 반복문 for
# for target in object (집합형 객체)
#    statement
# for i in [1,2,3,'good','nice']:
# for i in (1,2,3,'good','nice'):
for i in {1,2,3,'good','nice'}:
    print(i, end = ' ')
 
print()   
soft = {'java':'웹용언어', 'python':'만능언어','mysql':'dbms'}
for i in soft.items():
    print(i)
    print(i[0] + '^^;' + i[1])
    
print()
for k in soft.keys():
    print(k,end =' ')        

print()

for k in soft.values():
    print(k,end =' ') 
print()
    
for k,v in soft.items():
    print(k, ' ', v)
    
print()

li = ['a','b','c']
for i, d in enumerate(li): # enumerate <- index와 데이터를 반환하는 내장함수
    print(i, ' ', d)

print()
for n in [2, 3]:
    print('--{0}단--'.format(n))
    for i in[1,2,3,4,5,6,7,8,9]:
        print('{0}*{1}={2}'.format(n, i , n * i))
        
print()
datas = [1,2,3,4,5]
for i in datas:
    if i == 2:continue
    if i == 3:break
    print(i, end = ' ')
else:
    print('for 정상 종료일때 수행') #brake 의 비정상 종료이기 때문에 찍히지 않았음

print()
jumsu = [95, 73, 60, 50, 100]
number = 0 #합격자 번호
for jum in jumsu: #95,73,60,50,100 을 순서대로 jum에 대입
    number += 1
    if jum < 70:continue
    print('%d번째는 합격'%number)
    
print() 

li1 = [3 , 4, 5]
li2 = [0.5, 1, 2]
result = []
for a in li1:
    for b in li2:
        result.append(a + b)
print(result)

datas = [a + b for a in li1 for b in li2] #위에 코드를 한줄로 표현가능
print(datas)

print('단어의 발생 횟수를 dict로 저장--------')
import re

ss = """
삼성전자가 GAA(Gate-All-Around) 기술을 적용한 3㎚(나노미터·1㎚는 10억분의 1m) 반도체 양산을 세계 최초로 시작했다. 
반도체 파운드리(위탁 생산) 경쟁사인 대만 TSMC보다 빠른 양산이다. 
삼성전자의 파운드리 경쟁력이 한 단계 높아졌다는 평가가 나온다.
삼성전자는 30일 3㎚ 공정의 고성능 컴퓨팅(HPC)용 시스템 반도체를 초도 생산했다고 밝혔다. 
동시에 모바일 시스템온칩(SoC) 등으로 3㎚ 공정 적용을 확대해 나간다는 계획도 공유했다. 
3㎚ 공정은 반도체 제조 공정 가운데 가장 앞선 기술이다. 
반도체 초미세공정에서 나노 단위 공정은 성능을 결정하는 데 중요한 역할을 한다. 
반도체 초미세공정에서 나노 단위 공정은 성능을 결정하는 데 중요한 역할을 한다.
반도체 초미세공정에서 나노 단위 공정은 성능을 결정하는 데 중요한 역할을 한다.
얼마나 얇은 광원으로 정밀한 회로를 그릴 수 있느냐에 따라 성능이 정해지기 때문이다.
"""        

ss2 = re.sub(r'[^가-힣\s]','',ss) #한글,공백을 제외한 모든걸 삭제 앞에패턴을 ''로 치환한것
print(ss2)
ss3 = ss2.split(' ')
print(ss3)

cou = {}
for i in ss3:
    if i in cou:
        cou[i] += 1
    else:
        cou[i] = 1
print(cou)

print()
for ss in ['111-1234','일이삼-사오육칠','222-1234','2221234']:
    if re.match(r'^\d{3}-\d{4}$', ss):
        print(ss, '전화번호 맞네요')
    else:
        print(ss,'전화번호가 아니네요')
# print(sum([1, 5, 8]))   #합을구할때 쓰는 함수             
print('dict(사전형) 자료로 과일 값 출력 ---')
price = {'사과':2000, '감':500, '오렌지':1000} #시세
guest = {'사과':2, '감':3} #고객이 구매한 상품
bill = sum(price[f] * guest[f] for f in guest) #f를 순서대로 돌아 계싼함 price[0]->2000 guest[0]->2 2000*2=4000
print('고객이 구매한 과일 총액은 {0}원'.format(bill))
 
print()
datas = [1,2,'a',True,3] #list타입 모두 ok
li = [i * i for i in datas if type(i)==int]
print(li)

datas = {1,1,2,2,3} #set 타입 중복 x
se = {i *i for i in datas}
print(se)

id_name = {1:'tom', 2:'oscar'} #dict 형식 {'key':value...}
name_id = {val:key for key, val in id_name.items()}
print(name_id) #dict 형식으로 표출

print('-- 수열 생성 함수 : range() -----')
print(list(range(1, 6))) #1부터 6전까지 list 타입
print(tuple(range(1, 6))) #1부터 6전까지 tuble 타입 
print(set(range(6))) #0부터 6전까지 set 타입
print(list(range(1, 6, 2))) #1부터 6전까지 2씩 증가 list
print(list(range(-10, -100, -20))) #-10부터 -100전까지 -20씩 감소 list 타입

print()
for i in range(6):
    print(i, end =' ')
    
print()

for _ in range(3): #집합형 데이터를 어떤변수에 담을생각이 없을때 _
    print('hi')
    
for i in range(1,10): #구구단 2단 range로 만들기
    print('{}*{}={}'.format(2, i, i * 2))

tot = 0 #1~10까지 숫자의 합   
for i in range(1,11):
    tot +=i
    
print(tot)

print(sum(range(1,11))) #1~10까지 숫자의 합

print('문1: 2 ~ 9단까지 출력')
for i in range(2,10):
    for j in range(1,10):
        print('{}*{}={}'.format(i, j, i * j))

total=0       
print('문2: 1~ 100 사이에 정수중 3의 배수이면서 5의 배수의 합 출력')
for i in range(1,101):
    if i%3 ==0 and i%5 ==0:
        print(i)
        total=total+i
print('합은',total)        

print('문3: 주사위 눈금 합')
#주사위를 두 번 던져서 나온 숫자들의 합이 4의 배수가 되는 경우만 출력
# 예) 1,3 
# 예) 2,2

for i in range(1,7):
    for j in range(1,7):
        result=i+j
        if result%4 ==0:
            print(i,j)
 
print('\nN-gram : 문자열에서 N개의 연속된 요소를 추출하는 방법')
# 글자(문자) 단위 
text = 'Hello'
for i in range(len(text) - 1):
    print(text[i:i+2])
    
print()
text = 'this is python script'
words = text.split()
print(words)

for i in range(len(words) - 1):
    print(words[i],words[i+1]) 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
            
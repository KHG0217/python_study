a = 1

while a <= 5:
    print(a, end = ' ')
    a += 1

print()
i = 1
while i <= 3:
    j = 1
    while j <=4:
        print('i' +str(i)+ ',j' + str(j))
        j =j + 1
    i += 1    

print('1 ~ 100 사이 정수 중 3의 배수의 합')
i = 1; hap = 0
while i <= 100:
    # print(i)
    if i % 3 ==0:
        hap += i
    i += 1    
print('합' + str(hap))

print()

colors = ['빨강', '초록', '파랑'] #set 은 순서가 없기때문에 안됨 {}
a = 0
while a < len(colors):
    print(colors[a], end = ' ')
    a += 1

print()    

while colors:   #값이있으면 true 없으면 false기때문에 이렇게 작성해도 오류 x
    print(colors.pop())
    
print(len(colors))

print()

# n개당 별 n개찍
i = 1
while i <= 10:
    j = 1
    re = ''
    while j <= i:
        re = re + '*'
        j += 1
    print(re)
    i += 1
    
print('----------------------')
import time
# print(time.sleep(3)) #흐름을 멈추기, 자바에선 스레드

# sw = input('폭탄 스위치를 누를까요?[y/n]')
sw = 'n'
if sw =='Y' or sw =='y':
    count = 5
    while 1 <= count:
        print('%d초 남았어요'%count)
        print(time.sleep(1))
        count -= 1
    print('폭발!!!')
elif sw =='N' or sw == 'n':
    print('작업 취소')
else:
    print('y 또는 n을 누르시오')
    
print('-------continue, break------------')
a = 0

while a < 10: #숫자 0이면 거짓 그외 true
    a += 1
    if a == 5 :continue #가장 가까운 반복문으로 돌아감
    if a == 7 :break #무조건 탈출 else를 만나지 않음
    print(a)
else:
    print('while 정상 처리') #조건을 다 처리한 후에 진행함
    
print('while 수행 후 %d' %a)

print('컴이 가진 임의의 정수 맞추기 ---')
import random
# random.seed(42) #난수표에 42번줄 쭉 작성하여 고정시킴
# print(random.random()) # 0 ~ 1 실수로 난수를 발생시킴  
# print(random.randint(1, 100)) # 1 ~ 100 사이의 난수를 발생시킴
friend = ['tom' , 'james', 'oscar']
# print(random.choice(friend)) #1개만 뽑기
# print(random.sample(friend, 2)) #n개 뽑기
# random.shuffle(friend) #순서를 섞어줌 <-값 자체가 바뀜 in-place연산
# print(friend)

num = random.randint(1, 10)
while True:
    print('1 ~ 10 사이의 컴퓨터가 가진 예상 숫자를 입력:')
    guess = input()
    su = int(guess)
    if su == num:
        print("성공!"*5)
        break
    elif su < num:
        print('더 큰수를 입력하세요')
    elif su > num:
        print('더 작은수를 입력하세요')
        


    
        
            
          
                            
print('\n종료')

#문1) 1 ~ 100 사이의 숫자 중 3의 배수이나 2의 배수가 아닌 수를 출력하고, 합을 출력
i = sum = 0
while i< 100:
    if i % 3 == 0 and i % 2 !=0:
        print(i, end=',')
        sum += i
    i += 1                
print(sum) 

print()         

# 문2) 2 ~ 5 까지의 구구단 출력
i = 2;  
while i <= 5:
    index=1
    result=0
    while index <=9:
        result=i*index
        print('%d단'%i + ' ' + '%d'%i +'*'+'%d'%index + '=%d'%result, end=' '  )
        print()
        index += 1
    i += 1    
        
print()
        
# 문3) -1, 3, -5, 7, -9, 11 ~99 까지의 모두에 대한 합을 출력
i = -1; sum = 0
while -100<i<100:
    if i<0:
        sum += i
        i=-(i)+2
        
    else:
        sum += i
        i=-(i+2)        
print(sum)
       
print()   

         
        
         

#문4) 1 ~ 1000 사이의 소수 (1보다 크며 1과 자신의 수 이외에는 나눌 수 없는 수)와 그 갯수를 출력
i = 2
result=0
while i<=1000:
    count=0;
    j = 1; 
    while j <= i:        
        
        if i % j == 0:
            count +=1
        j += 1
        
        if count == 2:
           result += 1; 
    i += 1   
print(result) 

# 문4) 1 ~ 1000 사이의 소수(1보다 크며 1과 자신의 수 이외에는 나눌 수 없는 수)와 그 갯수를 출력
# i=2
# so=[]
# while i<=1000:
#     count = 0
#     j=1
#     while j<=i:
#         if i%j==0:
#             count+=1
#         j+=1
#     if count==2:
#         # print(i, end =' ')
#         so.append(i)
#     i+=1
# print()
#
# print(len(so))

    
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
    
    


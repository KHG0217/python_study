# 함수 : 여러 개의 수행문을 하나의 이름으로 묶어 놓은 실행단위
# 독립적으로 구성된 프로그램 코드의 집합
# 반복 코드를 줄여주며, 짧은 시간에 효과적인 프로그래밍 기능
# 내장함수, 사용자 정의함수
# 사용자정의함수 형식
# def 함수명(argument,...):
#    함수 내용...
# 클레스는 대문자 함수는 소문자로 쓰기로 약속

# 내장함수 일부 체험하기
print(sum({1,2,3}))
print(bin(8)) #2진수로 찍어라
print(int(1.7),float(3))
a = 10
b = eval('a + 5') #수식의 모양을 하고있는 문자열을 계산해주는 함수
print(b)

print(round(1.2), round(1.6)) #반올림

import math
print(math.ceil(1.2), math.ceil(1.6)) # 정수 근사치 중 큰 수
print(math.floor(1.2), math.floor(1.6)) # 정수 근사치 중 작은 수

print()
b_list = [True, 1, False]
print(all(b_list)) # 모두 참이면 참
print(any(b_list)) # 하나라도 참이면 참

b_list2 = [1,3,2,5,7,6]
result = all(a < 10 for a in b_list2)
print('모든 숫자가 10 미만?',result)

result = any(a < 3 for a in b_list2)
print('숫자 중 3미만이 있나요?',result)

print('복수개의 집합형 자료로 tuple 작성')
x = [10, 20, 30]
y = ['a', 'b']
for i in zip(x,y): #zip -> 쌍을 이루어주는 함수
    print(i)
    

    
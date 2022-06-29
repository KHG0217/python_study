#연산자
v1 = 3
v1 = v2 = v3 = 5
print(v1, v2, v3) #print는 라인스킵이다.
print('출력1', end=',') #기본은 줄바꿈이지만 이어서 쓰고싶다면 end속성 이용
print('출력2')
v1 = 1,2,3
print(v1) #() ->tuple
v1, v2 = 10, 20
print(v1, v2)
v2, v1= v1, v2
print(v1, v2)

print('값 할당 packing 연산')
v1, *v2= 1,2,3,4,5 # *packing 연산 1만 v1이 참조하고 나머지 v2가 참조
print(v1)
print(v2)

*v1, v2, v3= 1,2,3,4,5 # *packing 연산 1,2,3,4 v1이 참조하고 나머지 v2가 v3가 하나씩 참조
print(v1)
print(v2)
print(v3)

print('----------')
print(5 + 3, 5 - 3, 5 * 3, 5 / 3, 5 // 3, 5 % 3) # / ->실수나누기, //-> 정수나누기
print(divmod(5, 3)) #몫과 나머지를 한번에 얻기
a, b = divmod(5, 3)
print(a) #몫
print(b) #나머지

print(5 ** 3) #5를 3제곱

print('우선 순위: ', 3 + 4 * 5 , (3 + 4) * 5) # 소괄호 > 산술(*, / > +, -)> 관계 > 논리 > 치환
print(5 > 3, 5 == 3, 5!= 3)
print(5 > 3 and 4 < 3, 5 > 3 or 4 < 3, not(5 >= 3))

print()
print('한' + '국인' + ' '+ "파'이'팅" +' '+ '파"이"팅' )
print('한국인' * 10)

print('누적')
a = 10
a = a + 1
a += 1

# a++ 증감연산자는 파이썬에서 없다.
++a 
print('a : ', a)
print(a, a * -1, -a, --a, +a) #++,-- 는 +(+a), -(-a)

print()
print('bool 처리 :', True, False)
print(bool(True), bool(1), bool(-3.4), bool('a')) #어떤 값을 가지고있으면 true
print(bool(False), bool(0), bool(0.0), bool(''), bool([]), bool({}), bool(None)) #0이거나 값이 없으면 False

print('aa\nbb') #개행
print('aa\tbb') #tab키 누른것
print('aa\bbb') #빽스페이스
print(r'c:\aa\nbc') # 앞에 r(raw string)사용하여 이스케이프(escape) 기능을 해제한다.

print('print 관련 서식')
print(format(1.5678, '10.3f')) #소수 3자리까지 4째자리에서 반올림
print ('{0:.3f}'.format(1.0/3)) #소수 3자리까지 출력
print('나는 나이가 %d 이다.'%23) #정수
print('나는 나이가 %s 이다.'%'스물셋') #스트링
print('나는 나이가 %d 이고 이름은 %s이다.'%(23, '홍길동'))
print('나는 나이가 %s 이고 이름은 %s이다.'%(23, '홍길동'))
print('나는 키가 %f이고, 에너지가 %d%%.'%(177.7, 100))#실수, %붙으면 백분률
print('이름은 {0}, 나이는 {1}'.format('한국인', 33)) #인덱스로 넣기
print('이름은 {}, 나이는 {}'.format('신선해', 33)) #순서안주면 자동으로 순서에맞게 들어감
print('이름은 {1}, 나이는 {0}'.format(34, '강나루')) 

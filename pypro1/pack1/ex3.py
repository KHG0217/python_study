# 집합(묶음)형 자료형 : str ,list, tuple, set, dict

print('-----str------')
# str : 문자열 자료형. 순서가 있다.-> 인덱싱, 슬라이싱 가능, 수정 불가(int, float, complex, bool, str, tuple)

s = 'sequence'
print(len(s), s.count('e'))
print(s.find('e'), s.find('e', 3), s.rfind('e'))
# 다양한 문자열 관련함수들 검색을 통해 사용

ss = 'mbc' #일정기간동안 아무도 참조하지않는다 ? ->메모리 자동으로 해지 (가비지컬렉터)
print(ss, id(ss))
ss = 'abc'
print(ss, id(ss)) #주소가 다르다. 새로운 객체를 참조한것 수정된게 x

print()
print(s) #sequence
print(s[0], s[7]) # s[8] err 인덱싱
print(s[-1], s[-2]) #뒤에서부터 참조하는 것 인덱싱
print(s[0:5], s[:3], s[3:]) # 0이상5미만, 3미만, 3이상  슬라이싱 
print(s[-4:-1], s[-4:], s[1:8:2]) #1:8:step step =2개씩 띄어서 참조  슬라이싱

s = 'fre' + s[2:]
print(s)

sss='mbc sbs kbs'
imsi = sss.split(' ') #문자열 자르기 ex)' '공백으로 자르겠다.
print(imsi)
imsi2 = ','.join(imsi) #문자 합치기 ex),로 합치겠다.
print(imsi2)
aa = 'Life is too long'
bb = aa.replace('life', 'Your leg') #문자 치환하기 ex) life -> Your leg
print(bb)

print('-----list------') # 형식 list(), [요소...], 순서가 있다, 수정 가능, 중복 가능 다됨
a = [1,2,3]
print(a)
b = [10, a, 20.5, True, '문자열']
print(b, id(b))
print(b[0], b[1], b[1][2]) #인덱싱가능 ->슬라이싱도 가능하다.
print((b[2:5]))
print(b[-2::2]) #[-2]인덱스 이상 -> [-1] = True이상,[2]인덱스 미만 (20.5) 
print(b[1][:2])

print()
family = ['엄마','아빠','나']
family.append('남동생') #추가
family.insert(0, '할머니') #삽입
family.extend(['삼촌','고모']) #interable : 집합형 자료 (때거지로 쓸 수있다.) 많이 넣을 수있다.
family +=['아저씨','이모'] # +로도 넣어줄 수있다.
family.remove('나') #'나'가 삭제된다. (값에 의한 삭제)
del family[0] # family 0번째가 삭제된다.(순서에 의한 삭제)
print(family, len(family))

print('엄마' in family) #True
print('나' in family) #False

#주소값이 같고 요소값이 바뀜 =>수정이 가능하다.
aa = [1,2,3,1,2]
print(aa, id(aa))
aa[0] = 77
print(aa, id(aa))

print('자료구조 관련 : liFO') #셔틀콕 구조
kbs = [1,2,3]
kbs.append(4)
print(kbs)
kbs.pop() #꺼내다 -> 제일 위에(마지막것 부터) ex)여기선 4
print(kbs)
kbs.pop()
print(kbs)

print('자료구조 관련 : FiFO') #파이프 구조 
kbs = [1,2,3]
kbs.append(4)
print(kbs)
kbs.pop(0) #꺼내다 -> 제일 앞에(처음것 부터) ex)여기선 1
print(kbs)
kbs.pop(0)
print(kbs)

print('-----Tuple------') # 형식 tuple(), (요소...), List와 유사, 수정 불가능, 중복 가능, 대신 검색속도가 빠르다 (list와 비교했을때) read only!
# t = ('a', 'b', 'c', 'd')
t = 'a', 'b', 'c', 'd'
print(t, type(t), id(t), len(t), t.index('c'))

p = (1, 2, 3, 1, 2)
print(p, id(p), type(p))
# p[0] = 77 # err : 'tuple' object does not support item assignment

# 형변환 
p2 = list(p) #tuple -> list
print(p2, type(p2))
p3 = tuple(p2) #list -> tuple
print(p3, type(p3))


print('-----Set------') # 형식 set(), {요소...},순서가 없다. 수정 가능, 중복 안됨
a = {1,2,3,1,3}
print(a, type(a), len(a))

b = {3,4}
print(a.union(b)) #합집합
print(a.intersection(b))#교집합
print(a | b) #합집합
print(a - b) #차집합
print(a & b) #교집합
# print(b[0]) # TypeError: 'set' object is not subscriptable 인덱싱 불가
b.add(5) #추가
b.update({6,7}) #set 추가 
b.update([8,9]) #list 추가
b.update((10,11,12)) #tuple 추가
b.discard(8) #값에 의한 삭제
b.remove(9) #값에 의한 삭제
b.discard(8) #있으면 지우고 없으면 스킵
# b.remove(9) #있으면 지우고 없으면 에러
print(b)

li = [1,2,2,3,4,4,5,3]
print(li)
imsi = set(li) #set으로 중복 제거 
li = list(imsi) #list 형태로 바꾸기
print(li)

print('-----Dict------') # 형식 dict(), {'key':value, ...}, 순서 없다,수정 가능하다, key를 중복할 수 없다.
mydic = dict(k1=1, k2='abc',k3=1.2)
print(mydic, type(mydic), id(mydic))

dic = {'파이썬':'뱀', '자바':'커피', '스프링':'용수철'}
dic['여름'] = '장마철' #추가 여름이라는 key값에 장마철이라는 value넣기
print(dic, type(dic), len(dic))
del dic['여름'] #삭제
print(dic, type(dic), len(dic))
# dic.clear() #전부 삭제

dic['파이썬'] = '만능 언어' #key명 바꾸기
print(dic, type(dic), len(dic))
print(dic.keys()) # key만 찍기 , 반환값 list
print(dic.values())# value만 찍기 , 반환값 list
print(dic.get('파이썬'))# key값으로 value 뽑아내기.
print('파이썬' in dic) #dic에 '파이썬'이 있는지 확인 True
print('파이' in dic) #dic에 '파이'이 있는지 확인 False

for i in {1, 2, 3, 4, 5, 5, 5, 5}:
  print(i, end = ' ')

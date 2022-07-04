# 클래스는 새로운 타입을 만든다.
# Singer(가수) type을 만들고 여러 명의 가수 객체를 생성하기

class Singer:
    title_song = '화이팅 코리아 ~~'
    
    def sing(self):
        msg ='노래는 '
        print(msg, self.title_song,'랄~랄~라')

print('여러 명의 가수 객체를 생성 ---')
twice = Singer()
print(type(twice)) # Singer type
twice.sing()

twice.title_song = '우아하게' #twice의 title_song은 지역함수로 바뀌었다.
twice.sing()
twice.co = '(주)pjy'
print('소속사 :',twice.co)

bts = Singer()
print(type(bts))
bts.sing()
# print('소속사 :', bts.co) # AttributeError: 'Singer' object has no attribute 'co'

print(id(Singer), id(twice), id(bts))

print()
aaa = Singer # 주소를 치환
bbb = Singer() # Singer type의 객체 생성
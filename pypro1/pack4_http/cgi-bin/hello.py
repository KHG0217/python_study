# 파이썬 모듈(파일)로 출력값을 웹브라우저로 전송(java servlet을 생각)
# html - sql 사용 x, node.js ->javascript로 sql이 가능하게 해줌?, tensorflow
a = 10
b = 20
c = a + b
d = "결과는 " + str(c)

print('Content-Type:text/html;charset=utf8\n') #야 너한테 넘겨줄 내용이 text인데 html문서임 코드는 utf-8
print('<html><body>') #DOM - html ,xml (내용을 수정하고 추가,삭제해주는 것) -> 파이썬에선 bs가 해결해줌
print('<b>안녕하세요!</b> 파이썬 모듈로 작성한<br>문서입니다.')
print('<hr> 파이썬 변수 값 출력: %s'%(d,))
print('</html></body>') #w3c -> 웹표준화를 관리




#클라이언트가 전송한 값 처리 get 방식 데이터 받기 (java는 httpServletRequest)

import cgi

form = cgi.FieldStorage() #java로 httpServletRequest 역활 get방식으로 넘어온 데이터를 받는 역활

#변수(바뀌어도됨) = form[넘겨진명이랑 같아야함]
name = form['name'].value # java로 request.getParameter("name")
phone = form['phone'].value
gen = form['gen'].value

print('Content-Type:text/html;charset=utf8\n')
print('''
<html>

<body>
<h2>friend 문서 </h2>
이름은 {0}님, 전화번호는 {1}, 성별은 {2}
</body>
<br>
<a href='../index.html'>메인으로</a>
</html>

'''.format(name, phone,gen))
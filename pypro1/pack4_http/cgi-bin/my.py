#클라이언트가 전송한 값 처리 get 방식 데이터 받기 (java는 httpServletRequest)

import cgi

form = cgi.FieldStorage() #java로 httpServletRequest 역활 get방식으로 넘어온 데이터를 받는 역활

#변수(바뀌어도됨) = form[넘겨진명이랑 같아야함]
name = form['name'].value # java로 request.getParameter("name")
nai = form['age'].value

print('Content-Type:text/html;charset=utf8\n')
print('''
<html>

<body>
<h2>my </h2>
이름은 {0}님, 나이는 {1}세
</body>
<br>
<a href='../index.html'>메인으로</a>
</html>

'''.format(name, nai))
# 웹 서버 서비스 구현
from http.server import HTTPServer, CGIHTTPRequestHandler
#  CGIHTTPRequestHandler- 클라이언트와 서버사이에 데이터 주고받기,파이썬 파일 브라우저 출력용
# jsp나 servlet파일을 할 수 없다 - 톰켓이 없기떄문/ 그외 다 자바스크립트 뷰 등등 다 사용가능

# CGI(Common Gateway Interface) : 웹 서버와 외부 프로그램 사이에서 정보를 주고 받는 방법이나 규약
# 대화형 웹 페이지를 작성할 수 있게 된다.
class Handler(CGIHTTPRequestHandler):
    cgi_directories = ['/cgi-bin'] #여러개 줄 수있다.

serv = HTTPServer(('127.0.0.1',8889), Handler)

# GET /favicon.ico HTTP/1.1" 404 오류 - 페이지 상단에 이미지가 없다고 알려주는 것
print('웹 서버 서비스 시작...')
serv.serve_forever()
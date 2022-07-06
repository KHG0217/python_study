# 단순 서버 : 1회용
from socket import *

serversock = socket(AF_INET, SOCK_STREAM)   #socket(소켓종류, 소켓유향) 4버전, 파일
serversock.bind(('127.0.0.1',7788)) # ip,port 번호지정
serversock.listen(1) #클라이언트와 연결 정보 수 (리스너) = TCP의 listener 설정
print('server start ...')



conn, addr = serversock.accept()    #클라이언트 연결 대기
# conn: 클라이언트의 소켓과 addr: 클라이언트의 주소
print('client addr : ',addr)
print('from client msg : ', conn.recv(1024).decode()) #recv()클라이언트가 보내는 메시지를 읽고, decode() 디코딩으로 읽음 packet 단위(2진법)로 처리 되어 데이터를 넘겨줌
#바이너리 형태로 넘어옴(인코딩상태로) -> 디코딩
conn.closer()
serversock.close()
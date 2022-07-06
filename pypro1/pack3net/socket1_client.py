# 단순 클라이언트 : 1회용
from socket import *

clientsock = socket(AF_INET, SOCK_STREAM)
clientsock.connect(('127.0.0.1',7788)) # TCP 서버 연결을 시작한다.
clientsock.send('안녕 반가워'.encode(encoding='utf-8', errors='strict'))
#send()데이터 주고 .encode()인코딩 errors=?
clientsock.close()
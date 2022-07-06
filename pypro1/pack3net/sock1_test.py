# 네트워크를 위한 통신 채널 : socket
# socket: TCP/IP protocol의 프로그래머 인터페이스 이다.
# 프로세스 간에 대화가 가능하도록 하는 통신방식으로 client/server 모델에 기초한다.
# socket - socket이 정보를 교환한다.

import socket

print(socket.getservbyname('http', 'tcp')) #포트번호를 반환 http는 80을 쓰고있음
print(socket.getservbyname('telnet', 'tcp')) #23 원격접속 프로토콜 (잘 쓰진 않음)
print(socket.getservbyname('ftp', 'tcp')) # 21 파일전송 프로토콜
print(socket.getservbyname('smtp', 'tcp')) #25 메일 송수신 프로토콜
print(socket.getservbyname('pop3', 'tcp')) #110 이메일 수신 프로토콜
print()
print(socket.getaddrinfo('www.naver.com', 80, proto=socket.SOL_TCP))

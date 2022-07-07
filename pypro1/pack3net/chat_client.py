# 채팅 클라이언트
import socket
import threading
import sys

def handle(socket): #메세지 담당
    while True:
        data = socket.recv(1024)
        if not data: continue
        print(data.decode('utf-8'))
        
# 파이썬의 표준출력은 버퍼링이 됨 (출력내용이 버퍼에 계속 싸인다.)
sys.stdout.flush()  # buffer를 비우기

name = input('채팅 아이디 입력:')
cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cs.connect(('127.0.0.1', 5000))
cs.send(name.encode('utf-8'))

th = threading.Thread(target=handle,args=(cs, ))
th.start()

while True:
    msg = input() #채팅 메세지 입력
    sys.stdout.flush()
    if not msg: continue #msg가 없으면 넘김
    cs.send(msg.encode('utf-8')) #msg가 있으면 실행
    
cs.close()
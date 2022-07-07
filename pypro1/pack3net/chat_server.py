# 멀티 채팅 서버 : socket, thread
import socket
import threading

ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.bind(('127.0.0.1', 5000))
ss.listen(5)
print('채팅 서버 서비스 시작...')

#여러명에 대한 소켓 만들기
users = []
def chatUser(conn): 
    name = conn.recv(1024) #coon.name를 통해 msg를 받아옴
    data = '^^' + name.decode('utf-8') + '님 입장 ^^'
    print(data)
    
    try:
        for p in users:
            p.send(data.encode('utf-8'))
        while True:
            msg = conn.recv(1024) #coon.recv를 통해 msg를 받아옴
            suda_data = name.decode('utf-8') +'님 메세지: ' + msg.decode('utf-8')
            print(suda_data)
            
            #users 안에있는 사람들에게 반복문을 돌려 suda_data를 다시 보내는 코딩
            for p in users: 
                p.send(data.encode('utf-8'))
    #채팅한 사람이 나갔을때 오류        
    except:
        users.remove(conn) # 채팅방을 나간경우, users에서 socket 제거
        data = '~~' + name.decode('utf-8') +'님 퇴장하셨습니다.'
        print(data)
        if users:
            for p in users: #모든 사람들에게 퇴장했다고 보냄
                p.send(data.encode('utf-8'))
        else:
            print('exit')
    
while True:
    conn,addr = ss.accept() # 채팅을 원하는 컴퓨터가 접속한 경우 실행 클라이언트 소켓 conn이 받음
    users.append(conn) #user list에 넣기 
    th = threading.Thread(target=chatUser, args=(conn,))#멀티로 가능하게 하기위해 스레드 사용
    th.start()
    
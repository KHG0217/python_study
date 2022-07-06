# 단순 클라이언트 : 1회용
# cmd에서 접속하기
# 패키지 오른쪽설정 -> properties -> 경로복사
# cmd실행 -> cd(접속)띄고 복사한경로 -> dir(파일명보기) -> python(띄고)실행할 py명

from socket import *

clientsock = socket(AF_INET, SOCK_STREAM) #4버전 INET
clientsock.connect(('127.0.0.1',7788)) # TCP 서버 연결을 시작한다.
clientsock.send('안녕 반가워'.encode(encoding='utf-8', errors='strict'))
#send()데이터 주고 .encode()인코딩 errors=?
clientsock.close()
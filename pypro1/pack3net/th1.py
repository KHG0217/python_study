# Thread : lightweight process 라고도 한다. 작업 실행단위를 말함
# process 내에서 여러개의 스레드를 운영해 멀티 태스킹을 할 수 있다. 메모리를 공유.
# 기본적으로 하나의 프로그램은 하나의 스레드(메인 스레드) 를 가지고 있다.

import threading, time

def myrun(id):
    for i in range(1, 11):
        print('id={}-->{}'.format(id,i))
        time.sleep(0.3)

# 스레드를 사용하지 않은 경우
# myrun(1)
# myrun(2)

# 스레드를 사용하는 경우
# threading.Thread(target='수행함수명')
th1 = threading.Thread(target=myrun, args=('일')) #사용자 정의 스레드 1, args-> myrun에 들어갈 아규먼트
th2 = threading.Thread(target=myrun, args=('이')) #사용자 정의 스레드 2
th1.start() #스레드 시작
th2.start() #스레드 시작
#메인스레드,스레드1,스레드2 총 3개의 스레드가 수행 (동시에는 아님)

th1.join()
th2.join()
# join() 사용자 정의 스레드가 끝나기 전까지 메인스레드의 실행을 멈춤
print('프로그램 종료')



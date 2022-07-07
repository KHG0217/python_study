# 스레드 간 공유 자원값 충돌 방지

import threading, time

g_count = 0 # 전역변수는 자동으로 스레드의 공유자원이 된다.
lock = threading.Lock()

def threadCount(id, count):
    global g_count
    
    for i in range(count):
        lock.acquire() # 두 개 이상의 스레드 간 충돌 방지를 위한 lock이 걸린다.
        #                하나의 스레드가 공유자원(여기선 g_count)을 쓰고 있을때 다른 스레드는 대기 상태가 된다.
        print('id %s ==> count:%s, g_count:%s'%(id,i,g_count))
        g_count += 1
        lock.release() # lock 해제

for i in range(1, 6):
    threading.Thread(target=threadCount, args=(i, 5)).start()

#id 3 ==> count:3, g_count:13id 4 ==> count:0, g_count:13 공유자원 충돌 :g_count
#충돌 방지

time.sleep(1)   
print('최종 g_count :', g_count)
print('프로그램 종료')
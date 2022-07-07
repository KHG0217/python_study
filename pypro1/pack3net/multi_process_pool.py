# 파이썬은 thread는 GIL(Global Interpreter Lock)정책을 따르고 있다.
# 그래서 완벽한 병렬처리를 지원하지 못함.
# 이런 문제를 해결하기 위한 multiprocessing을 지원하는 모듈이 제공된다.
# multiprocessing은 비동기적이고, 부분 작업이 비결정적인 경우에 효과적이다

# 비결정적 -> 무작위하고 랜덤하다.
# 비동기적 ->어떤 작업을 요청했을 때 그 작업이 종료될때 까지 기다리지 않고 다른 작업을 하고 있다가,
#         요청했던 작업이 종료되면 그에 대한 추가 작업을 수행하는 방식

# 주로 네트워킹 작업에서 효과적

# Pool 클래스 사용
# 입력값에 대해 process들을 한개씩 건너건너 분해하여 함수 실행을 병렬화 해줌

from multiprocessing import Pool
import time
import os

def func(x):
    print('값',x,'에 대한 작업 pid(process id): ', os.getpid()) # 현재 프로세스 아이디를 확인
    time.sleep(1)
    return x*x
if __name__ =='__main__':
    startTime = int(time.time())
    """
    for i in range(1, 11): # 실습1 : 일반적인 방법으로 함수 호출 반복
        #process id를 하나로 처리하고 있음을 알 수 있다.
        print(func(i)) 
    """
    
    # 방법2 : Pool 사용
    pool = Pool(3)  # process를 세 개를 준비 (3 ~ 5개가 적당)
    print(pool.map(func,range(1,11))) # .map 함수와 인자값을 맵핑하는 함수
    #3개 씩 묶어서 처리됨을 알 수 있다.
    endTime = int(time.time())
    
    print('총 소요 시간: ',(endTime - startTime))
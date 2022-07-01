# 현재 모듈은 다른 모듈에서 필요한 멤버를 적어 놓은 파일이다.
name = '파이썬 만세'

def ListHap(*ar):
    print(ar)
    
    if __name__ == '__main__':   #이곳이 메인모듈인지 확인하는 코드
        print('여기가 최상위 모듈')
    
def Kbs():
    print('대한민국 대표방송')
    
def Mbc():
    print('문화 방송 채널 11')
    
    
# 예외처리 : 프로그램 진행 도중 발생되는 에러에 대한 처리 방법 중 하나
# try ~ except

def divide(a, b):
    return a / b

# c = divide(5, 2)
# c = divide(5, 0) # ZeroDivisionError: division by zero

# print('c : ', c)

try:        #java는 try ~ catch =외부 장치하고 묶어쓸때(키보드 입력,네트워크작업 등) 반드시 예외처리
    c = divide(5, 2)
    # c = divide(5, 0)
    print('c : ', c)
    
    mbc = [1, 2]
    print(mbc[0])
    # print(mbc[2]) #IndexError: list index out of range
    
    f = open('c:/work/aa.txt') #FileNotFoundError: [Errno 2] No such file or directory: 'c:/work/aa.txt'

    
except ZeroDivisionError:   # 0을 만나면 
    print('두번째 인자로 0을 주지 마세요')
except IndexError as err:       #인덱스 오류
    print('참조범위 오류: ', err)
except Exception as e:          #나머지 오류
    print('기타 나머지 오류: ', e)
finally:
    print('에러 유무와 상관없이 반드시 수행되는 부분')
    
print('프로그램 종료')



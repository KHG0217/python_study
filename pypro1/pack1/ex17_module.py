# 모듈(Module) : 소스 코드의 재사용을 가능하게 함
# 소스 코드를 하나의 이름 공간으로 구분하고 관리함
# 하나의 파일은 하나의 모듈이다.
# 표준 모듈, 사용자 정의 모듈, 제3자(Third party) 모듈
# 표준 모듈: 파이썬을 설치하면 이미 만들어져 있는 모듈
# 사용자 정의 모듈: 만든 모듈
# 제3자 모듈: 전문가가 만들어 놓은 모듈을 다운로드받아서 불러쓰기

# 내장된 모듈(표준 모듈) 읽어 사용하기
a = 10
print(a)

import sys
print('모듈 경로: ', sys.path)
# sys.exit() #프로그램 강제종료 java= system.exit()와 같음
print('종료')

import math
print(math.pi)
print(math.sin(math.radians(30)))

import calendar
calendar.setfirstweekday(6)
calendar.prmonth(2022, 7)

print(dir(calendar)) #dir(모듈) = 사용가능한 모듈리스트

#...

print('난수 출력')
#import 모듈명
import random
print(random.random())
print(random.randrange(1, 10, 1))

#from 모듈명 import 멤버
from random import randrange, randint #메모리에 로딩해놓고 쓰는것
print(randrange(1, 10, 1))
print(randint(1,10))

from random import * #random멤버를 전부 로딩해놓고 쓰는것 권장 x
print(randint(1,10))



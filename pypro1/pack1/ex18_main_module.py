# 사용자 정의 모듈 작성 후 읽기
print('현재 모듈에서 뭔가를 다른 모듈 호출하기')
a = 10
print('a :', a)
print(dir()) #기본모듈을 지원하고있는 멤버

print()
import pack1.ex18_mymod1 #패키지명.모듈명

list1 = [1,3]
list2 = [2,4]
pack1.ex18_mymod1.ListHap(list1, list2) #패키지명.모듈명.멤버


def abc():
    if __name__ == '__main__':   #이곳이 메인모듈인지 확인하는 코드
        print('여기가 최상위 모듈이라고 외칩니다')
    
abc()
pack1.ex18_mymod1.Kbs()

from pack1 import ex18_mymod1
ex18_mymod1.Kbs()

from pack1.ex18_mymod1 import Kbs, Mbc , name
Kbs()
Mbc()
print(name)

print()
# 다른 패키지의 모듈 읽기
from pack_other import ex18_mymod2
ex18_mymod2.Hap(5, 3)

import pack_other.ex18_mymod2
ex18_mymod2.Cha(5, 3)

print()
# path가 설정된 지역의 모듈 호출
import ex18_mymod3 #path에 넣어두어서 모듈명만 써도 import가 됨
ex18_mymod3.Gop(5, 3)

from ex18_mymod3 import Nanugi
Nanugi(5, 3)

from ex18_mymod_test import test
test(1, 2)


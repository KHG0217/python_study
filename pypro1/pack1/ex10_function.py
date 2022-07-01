# 함수 연습용 게임
import random
import time

def disp_intro():
    print('보물이 가득한 두개의 동굴 속에 두 마리 용이 있다. 착한 놈/무서운 놈')
    print('번호를 선택해 부자가 되느냐 아니면 ...')
    
def choose_cave():
    cave = ''
    while cave != '1' and cave != '2':
        print('동굴 번호를 산택하라(1 or 2)')
        cave = input()
    return cave

def chk_cave(number):
    print('동굴에 도착 와우 떨린다. (좌우 살핌)')
    time.sleep(3)
    print()
    ranNum = random.randint(1, 2)
    
    if number == str(ranNum):
        print('아싸 착한용을 만나 부자가 됨')
    else:
        print('그를 본 사람은 아무도 없었다.')
        
play_again = 'y'

while play_again == 'y':
    disp_intro()
    caveNumber = choose_cave()
    chk_cave(caveNumber)
    print('계속 할까요?(y/n)')
    play_again = input()
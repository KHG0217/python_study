# 동(리) 이름을 입력해서 해당 동(리)이름으로 시작되는 우편번호 자료 출력

def abc():
    try:
        dong = input('동(면)이름 입력:')
        # dong = '개포'
        print(dong)
        
        with open(r'zipcode.txt', mode='r', encoding='euc-kr') as f:
            # line = f.read() #전체 자료 읽기
            line = f.readline() # 한 행 읽기
            # lines = line.split('\t')
            lines = line.split(chr(9)) #아스키 코드로 자르기 아스키코드 9 =tap 그외 :10+13 = 엔터키 
            # print(lines) #['135-806', '서울', '강남구', '개포1동 경남아파트', '', '1\n']
            # print(lines)
            
            while line: #line에 값이 있으면 true 없으면 false
                # lines = line.split('\t') #tap 키 = \t
                lines = line.split(chr(9))
                if lines[3].startswith(dong): #3번째의 앞에 글자가 dong[개포]과 같냐
                                                                #\ <- 한줄로 이어지고있다고 알려줌
                    print('['+ lines[0] + ']' + lines[1] + ' '+ \
                          lines[2]+ ' ' + lines[3] + ' ' + lines[4])
                line = f.readline() #다음 한 행 읽기
                
    except Exception as e:
        print('err: ', e)
        
if __name__=='__main__':
    abc() 
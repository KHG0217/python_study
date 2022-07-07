# 멀티 프로세싱을 이용한 웹 스크래핑 : 여기선 소요시간 확인에 관심을 갖는다.
from bs4 import BeautifulSoup as bs
import time
import requests
def get_link():
    data = requests.get('https://beomi.github.io/beomi.github.io_old/').text
    # print(data)
    soup = bs(data, 'http.parser') #읽어들인 데이터에 대해서 , parser 파싱한다.
    # print(soup,type(soup))
    my_title = soup.select('h3 >a') #h3의 자식인 a요소 가져온다.
    # print(my_title)
    data = []
    
    for t in my_title:
        data.append(t.get('href')) #href 속성에 있는 값만 가져와서 넣는다.
    return data

def get_contents(link):
    # print(link)
    abs_link = 'https://beomi.github.io' + link
    # print(abs_link) # 각 페이지에 접근할 수 있는 url을 완성
    
    data = requests.get(abs.link).text
    soup = bs(abs_link, 'hml.parser')
    print(soup.select('h1')[0].text) #h1 속성중첫번째만 출력
    print('*********'*50)

if __name__ =='__main__':
    start_time = time.time()
    
    for link in get_link():
        get_contents(link)
        
    
    print('--%s 초 소요됨'%(time.time() - start_time))
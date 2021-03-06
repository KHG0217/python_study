# 멀티 프로세싱을 이용한 웹 스크래핑 : 소요시간 확인에 관심
import requests
from bs4 import BeautifulSoup as bs
import time
from multiprocessing import Pool

def get_link():
    data = requests.get('https://beomi.github.io/beomi.github.io_old/').text
    # print(data)
    soup = bs(data, 'html.parser')
    # print(soup, type(soup))
    
    my_title = soup.select('h3 > a') #h3의 자식요소로 a 인것을 셀렉
    # print(my_title)
    data = []
    
    for t in my_title:
        data.append(t.get('href')) #my_title중 href요소 들어간것 data에 넣기
    
    return data
    

def get_contents(link):
    # print(link)
    abs_link = 'https://beomi.github.io' + link
    # print(abs_link)  # 각 페이지에 접근할 수 있는 url을 완성
    
    data = requests.get(abs_link).text
    soup = bs(data, 'html.parser')
    print(soup.select('h1')[0].text)
    print('*****' * 20)


if __name__ == '__main__':
    start_time = time.time()
    '''
    #실습1 : 멀티 프로세싱 X
    for link in get_link():
        get_contents(link)
    '''
    
    # 실습2: 멀티 프로세싱 O
    
    pool = Pool(processes = 4)
    pool.map(get_contents, get_link())
        
    print('--%s 초 소요됨'%(time.time() - start_time))
    
    print('')
    print('방문 사이트 총 수:', len(get_link()))
    
    
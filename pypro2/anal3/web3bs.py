# BeautifulSoup은 HTML 및 XML 파일에서 데이터를 가져오기 위한 DATA 라이브러리이다.

import requests  

from bs4 import BeautifulSoup

def go():
    base_url = "http://www.naver.com:80/index.html"

    #storing all the information including headers in the variable source code
    source_code = requests.get(base_url)

    #sort source code and store only the plaintext
    plain_text = source_code.text   
    print(plain_text)
  
    
    #converting plain_text to Beautiful Soup object so the library can sort thru it
    convert_data = BeautifulSoup(plain_text, 'lxml')  # (xml or 문서파일, 'lxml')
    for link in convert_data.findAll('a'):
        href = base_url + link.get('href')  #Building a clickable url
        print(href)                          #displaying href

    

go()
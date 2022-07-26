# 형태소 분석 : 자연어를 형태소를 비롯하여 어근, 접두사, 접미사, 품사등 다양한 언어적 속성의 구조를 파악.
# pip install JPype1
# 영문 : NlTK
# 한글 : KONLPY
from konlpy.tag import Kkma, Okt, Komoran

kkma = Kkma()
print(kkma.sentences('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다')) # 문장별
print(kkma.nouns('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다')) # 명사만
print(kkma.morphs('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다')) # 모든품사
print()

okt = Okt()
print(okt.nouns('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(okt.morphs('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(okt.pos('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다')) # 품사테그와 함께나옴
print(okt.pos('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다', stem=True)) #원형 어근 출력
print(okt.phrases('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다')) # 어절 단위
print()

ko =Komoran()
print(ko.nouns('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(ko.morphs('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(ko.pos('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
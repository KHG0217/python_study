# 개인용 RDBMS - sqlite3 : 파이썬에 기본모듈로 제공
# 데이터를 저장할때 - file, database
# file - 필요한정보를 찾기가 번거러움 ->database

import sqlite3
print(sqlite3.sqlite_version)

print()
conn = sqlite3.connect('example.db') # 데이터베이스 생성
# conn = sqlite3.connect(':memory:') # 테스트 용 - 주 기억장치(RAM)에 저장됨 (휘발성) = 실행될 때만 있고 안하면 사라짐 

try:
    # 커서 객체로 SQL문 처리
    cursor = conn.cursor()
    
    # table 작성    # sqlite3 type : integer, real, text, blob
    cursor.execute("create table if not exists friend(name text, phone text, addr text)") # 원래 SQL문은 대문자, 근데 자동으로 대문자로 변환해서 들어감
    
    # insert data
    cursor.execute("insert into friend(name,phone,addr) values('한국인','111-1111','역삼1동')")
    cursor.execute("insert into friend(name,phone,addr) values('신기해','222-1111','역삼2동')")
    input_data = ('조조','333-1111','서초2동')
    cursor.execute("insert into friend(name,phone,addr) values(?,?,?)", input_data)
    conn.commit()
    
    # select data
    cursor.execute("select * from friend")
    # print(cursor.fetchone()) #데이터를 하나씩 읽는것.
    print(cursor.fetchall())
    
    print()
    cursor.execute("select name,addr,phone from friend")
    for c in cursor:
        print(c[0] +' '+ c[1] + ' '+ c[2])
    
except Exception as e:
    print('err : ', e)
    conn.rollback()
finally:
    conn.close() #메모리 해제 (가비지 컬렉터에게 메모리 해제해달라고 요청하는 것)

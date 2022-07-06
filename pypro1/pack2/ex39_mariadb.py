# 원격 데이터버베이스 서버(MariaDB)와 연동
# pip install mysqlclient 로 드라이브 파일 설치 (아나콘다 프롬프트)

import MySQLdb
"""
conn = MySQLdb.connect(host = '127.0.0.1', user = 'root', port = 3306,#port = 기본값 3306(생략가능) 기본 3306이라면 생략가능
                       password='maria123', database='test') #mariadb에 접속하기, dict타입

print(conn)
conn.close()
"""
config = {

    'host':'127.0.0.1',

    'user':'root',

    'password':'maria123',

    'database':'test',

    'port':3306,

    'charset':'utf8',

    'use_unicode':True

} #dict 타입 

try:
    conn = MySQLdb.connect(**config) #아규먼트에 dict 타입을 원하므로 **
    cursor = conn.cursor()
    
    #자료 추가
    # sql = "insert into sangdata(code,sang,su,dan) values(10,'상품1',5,1000)"
    # cursor.execute(sql)
    # conn.commit() # 파이썬은 반드시 커밋을 해줘야한다. !java는 오토커밋
                #?,?,?,? 자리에 %s
    """ 
    sql = "insert into sangdata values(%s,%s,%s,%s)"
    sql_data = ('11','상품2',12,2000) #tuple로 보내줌 () 안써도 tuple
    cou = cursor.execute(sql,sql_data) #sql 에 sql_data를 1:1로 맵핑해줌
    conn.commit()
    print('cou :', cou)
    if cou ==1:
        print('추가성공')
    """
    """
    # 자료 수정
    sql = "update sangdata set sang=%s,su=%s,dan=%s where code=%s"
    sql_data=('파이썬',7,5000,10)
    cursor.execute(sql,sql_data)
    conn.commit()
    """
    """
    # 자료 삭제
    code = '10'
    
    #비권장 secure coding guideline에 위배 sql injection 해킹 위험
    # sql = "delete from sangdata where code=" + code 
    
    #sql = "delete from sangdata where code='{0}".format(code) # 권장 1
    sql = "delete from sangdata where code=%s" #권장 2
    cursor.execute(sql,(code,)) #(code,) <-반드시 tuple형식으로 넣어야한다
    conn.commit()
    """
    
    #자료 읽기
    sql = "select code, sang, su, dan from sangdata"
    cursor.execute(sql)
    
    for data in cursor.fetchall():
        # print(data)
        print('%s %s %s %s'%data)
        
    """    
    print()
    for r in cursor:
        # print(r)
        print(r[0], r[1], r[2], r[3])
        
    print()
    for (code, sang, su, dan) in cursor:
        print(code, sang, su, dan)
    
    print()
    for (a, b, c, d) in cursor:
        print(a, b, c, d)
    """
except Exception as e:
    print('에러: ',e)
finally:
    cursor.close()
    conn.close()
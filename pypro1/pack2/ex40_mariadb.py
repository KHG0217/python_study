# 키보드로 부서번호를 입력받아 해당 부서에 근무하고 있는 직원 출력 

import MySQLdb

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
"""

import pickle
with open("mydb.dat", mode='rb') as obj:
    config = pickle.load(obj)

def chulbal():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        
        buser_no = input('부서번호 입력') #sql문을 먼저 프롬프트에서 확인해보는걸 추천
        sql = """
            select jikwon_no,jikwon_name,buser_num,jikwon_pay,jikwon_jik
            from jikwon
            where buser_num={0}
        """.format(buser_no)
        # print(sql)
        
        cursor.execute(sql)
        datas = cursor.fetchall()
        # print(datas,len(datas)) #len():갯수
        if len(datas) == 0:
            print(str(buser_no) + '번 부서는 없어요')
            return  #    sys.exit() <- 프로그램의 종료
        
        for  jikwon_no,jikwon_name,buser_num,jikwon_pay,jikwon_jik in datas:
            print( jikwon_no,jikwon_name,buser_num,jikwon_pay,jikwon_jik)
            
        print('인원 수:' + str(len(datas)))
    except Exception as e:
        print('err : ', e)
    finally:
        cursor.close()
        conn.close()
        
if __name__ =='__main__':
    chulbal()
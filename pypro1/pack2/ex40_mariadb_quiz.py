import pickle
import MySQLdb
with open("mydb.dat", mode='rb') as a:
    config = pickle.load(a)
    
def check():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        
        num = input("사원번호 입력")
        name = input("직원명")
        sql1 ="""
            select jikwon_no,jikwon_name
            from jikwon
            where jikwon_no={0} and jikwon_name='{1}'
        """.format(num,name)
        cursor.execute(sql1)
        datas1 = cursor.fetchone()
        
        if len(datas1) == 0:
            print('직원명과 직원번호가 일치하지 않습니다.')
        else:
            print('로그인 성공')                  
            sql2 ="""
                select gogek_no,gogek_name,gogek_tel
                from jikwon,gogek
                where jikwon_no={0} and gogek_damsano={0}
            """.format(num)
            cursor.execute(sql2)
            datas2 = cursor.fetchall()
            members=len(datas2) 
            for num,name,tel in datas2:
                print('고객번호:{0}, 고객이름:{1}, 고객번호:{2}'.format(num,name,tel))
            print()
        
            print('관리 인원수: ',members )

            
    except Exception as e:
        print('err :',e)
        
    finally:
        cursor.close()
        conn.close()
if __name__ =='__main__':
    check()        
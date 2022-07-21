# 원격 DB (MariaDB) : jikwon 테이블

import MySQLdb
import pickle
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family = 'malgun gothic') # 그래프에 한글 깨질 때
plt.rcParams['axes.unicode_minus'] = False # 한글깨짐 방지후 음수깨짐 방지

try:
    with open('mydb.dat', mode = 'rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('읽기 오류:', e)

try:
    conn = MySQLdb.connect(**config) # ** dict 타입
    cursor = conn.cursor()
    sql = """
        select jikwon_no,jikwon_name,buser_name,jikwon_jik,jikwon_gen,jikwon_pay
        from jikwon inner join buser
        on buser_num = buser_no
    """
    
    cursor.execute(sql)
    
    # 1)  console로 출력
    # for(a,b,c,d,e,f) in cursor:
    #     print(a,b,c,d,e,f)
    
    # 2) DataFrame으로 출력
    df1 = pd.DataFrame(cursor.fetchall(),
                        columns=['jikwon_no','jikwon_name','buser_name','jikwon_jik','jikwon_gen','jikwon_pay']
                       )
    print(df1.head(3))
    print()
    
    # 3) csv 파일로 출력
    import csv
    #    피클로 객체 저장하기
    with open("jikwon_data.csv", mode= 'w', encoding='UTF-8') as fobj:
        w = csv.writer(fobj)
        for r in cursor:
            w.writerow(r)
            
    # csv 파일 읽기
    df2 = pd.read_csv("jikwon_data.csv",header=None,
                      names=['번호','이름','부서','직급','성별','연봉'])
    print(df2.head(3))
    print()
    
    print('pandas의 sql 처리 기능 사용해서 읽기')
    df = pd.read_sql(sql,conn)
    df.columns = ['번호','이름','부서','직급','성별','연봉']  
    print(df.head(3))
    
    print('기술통계 : 자료 요약 정리. 평균, 분산, 산포도, 분포, 요약 통계량, 도수분포표, 외도, 첨도, 시각화.... ------')
    print(df[:3])
    print(df[:-27])
    
    print('건수 : ', len(df), df['이름'].count())
    print('부서별 인원수 : ', df['부서'].value_counts())
    print('연봉 평균 : ', df.loc[:,'연봉'].mean())
    
    # 빈도수
    ctab = pd.crosstab(df['성별'],df['직급'])
    print(ctab)
    print()
    
    print(df.groupby(['성별','직급'])['이름'].count()) #성별,직급별 인원수(이름.count)
    print()
    
    print(df.pivot_table(['연봉'], index=['성별','직급'], aggfunc=np.mean)) # 성별,직급별 연봉의 평균
    print()
    
    # 시각화: pie - 직급별 연봉 평균
    jik_ypay  = df.groupby(['직급'])['연봉'].mean() #직급별 연봉평균
    print(jik_ypay, type(jik_ypay)) # <class 'pandas.core.series.Series'> 인덱스 과장.. 벨류 수치
    
    plt.pie(jik_ypay, explode=(0.2,0,0,0,0.3), labels=jik_ypay.index,
            shadow=True, labeldistance=0.7, counterclock=False)
    plt.show()
    
    
except Exception as e:
    print('읽기 오류:', e)    
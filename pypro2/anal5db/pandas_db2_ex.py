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
    df = pd.read_sql(sql,conn)
    df.columns = ['번호','이름','부서','직급','성별','연봉']
    
    print(df)
    
    # print(df.pivot_table(['연봉'], index=['성별'], aggfunc=np.mean))
    data = df.pivot_table(values=['연봉'], index=['성별'], aggfunc=np.mean)

    plt.bar(data['연봉'].index,data['연봉'].values,color=['black','red'])
    plt.xlabel('성별')
    plt.ylabel('연봉')
    plt.xticks(data['연봉'].index, labels=['Man','Woman'])
    plt.show()
    
    
    # data_result = pd.merge(df, pop_seoul, on='구별')
    ctab = pd.crosstab(df['성별'],df['부서'])
    print(ctab)
    ctab.to_html
    
    
    
  
except Exception as e:
    print('읽기 오류:', e)     
    
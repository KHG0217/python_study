import pandas as pd
import scipy.stats as stats
import MySQLdb
import pickle
import  numpy as np
from bokeh.layouts import column


# 카이제곱 문제1) 
# 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
#
# 귀무가설: 부모의 학력수준과 자녀의 대학 진학여부는 관련이 없다.
# 대립가설: 부모의 학력수준과 자녀의 대학 진학여부는 관련이 있다.
#   예제파일 : cleanDescriptive.csv
#
#   칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
#
#   조건 :  level, pass에 대해 NA가 있는 행은 제외한다.



data = pd.read_csv('testdata/cleanDescriptive.csv').dropna(subset=['level', 'pass'],axis=0)
# print(data['level'])

ctab =pd.crosstab(index = data['level'], columns=data['pass'])


ctab.index = ['고졸','대졸','대학원졸']
ctab.columns = ['진학x','진학']
# print(ctab)

chi2, p,ddof,exp = stats.chi2_contingency(ctab)
print('chi2:{}, p-value:{}, df:{}'.format(chi2,p,ddof))
# chi2:2.7669512025956684, p-value:0.25070568406521365, df:2

# 해석: p-value 값이 0.05 이상 이므로 귀무가설 채택
#     부모의 학력수준과 자녀의 대학 진학여부는 관련이 없다.



# 카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 

# 귀무가설: 직급과 연봉은 관련이 없다.
# 대립가설: 직급과 연봉은 관련이 없다.
#
# 그렇다면 정말로 jikwon_jik과 jikwon_pay 간의 관련성이 있는지 분석. 가설검정하시오.
#
#   예제파일 : MariaDB의 jikwon table 
#
#   jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#
#   jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
#
#   조건 : NA가 있는 행은 제외한다

try:
    with open('mydb.dat', mode = 'rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('오류: ',e )
    
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql ="""
        select jikwon_jik,jikwon_pay from jikwon
    """
    cursor.execute(sql)
    
    df = pd.DataFrame(cursor.fetchall(),
                       columns=['직급','연봉']
                      )
    print(df)
    # df['직급'] = np.where(df['직급'] == '이사', 1, df['직급'])
    # df['직급'] = np.where(df['직급'] == '부장', 2, df['직급'])
    # df['직급'] = np.where(df['직급'] == '과장', 3, df['직급'])
    # df['직급'] = np.where(df['직급'] == '대리', 4, df['직급'])
    # df['직급'] = np.where(df['직급'] == '사원', 5, df['직급'])
    df['직급'] = df['직급'].apply(
        lambda g:1 if g == '이사' else 2 if g == '부장' else 3 \
                    if g == '과장' else 4 if g == '대리' else 5)
    
    df['연봉'] = np.where(df['연봉'] < 3000, 1, df['연봉'])
    df['연봉'] = np.where((df['연봉'] >= 3000) & (df['연봉'] < 5000), 2, df['연봉'])
    df['연봉'] = np.where((df['연봉'] >= 5000) & (df['연봉'] < 7000), 3, df['연봉'])
    df['연봉'] = np.where(df['연봉'] >= 7000, 4, df['연봉'])
    
    ctab2 = pd.crosstab(index = df['연봉'], columns= df['직급'], dropna=True)
    print(ctab2)

    
    
    
    chi3, p2,ddof2,exp = stats.chi2_contingency(ctab2)
    print('chi2:{}, p-value:{}, df:{}'.format(chi3,p2,ddof2))
    # chi2:37.40349394195548, p-value:0.00019211533885350577, df:12
    
    # 해석: p-value 값이 0.05 이하 이므로 귀무가설 기각 대립가설 채택
    #     직급과 연봉은 관련이 있다.
    


    
    
    
    
    

except Exception as e:
    print('오류 : ', e)







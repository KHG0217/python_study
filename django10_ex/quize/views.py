
from datetime import date
import MySQLdb
from django.shortcuts import render

import matplotlib.pyplot as plt
import pandas as pd
from quize.models import Jikwon


# config = {
#     'host':'127.0.0.1',
#     'user':'root',
#     'password':'maria123',
#     'database':'test',
#     'port':3306,
#     'charset':'utf8',
#     'use_unicode':True
# }
# plt.rc('font', family='malgun gothic')
# # Create your views here.
# def listFnc(request):
#     conn = MySQLdb.connect(**config)
#     cursor = conn.cursor()
#     sql ='''
#         select jikwon_no,buser_no,jikwon_name,buser_name,jikwon_jik,jikwon_pay,jikwon_ibsail
#         from jikwon,buser
#         where jikwon.buser_num = buser.buser_no
#         order by buser_no, jikwon_name asc
#      '''
#     cursor.execute(sql)
#     rows = cursor.fetchall()
#     df= pd.DataFrame(rows)
#     df.columns=['사번','부서번호','직원명','부서명','직급','연봉','입사']
#
#
#     jik_group = df['연봉'].groupby(df['직급'])
#     jik_group_detail = {'sum':jik_group.sum(),'avg':jik_group.mean()}
#     df2 = pd.DataFrame(jik_group_detail)
#     df2.columns =['연봉합','연봉평균']
#
#     jik_group2 = df['연봉'].groupby(df['부서명'])
#     jik_group_detail2 = {'sum':jik_group2.sum(),'avg':jik_group2.mean()}
#     df3 = pd.DataFrame(jik_group_detail2)
#     df3.columns =['연봉합','연봉평균']
#
#     jik_result = jik_group2.agg(['sum', 'mean'])
#     jik_result.plot(kind ='bar')
#     plt.title('부서별 연봉합과 평균')
#     plt.xlabel('연봉')
#     fig = plt.gcf()
#     fig.savefig('django10_ex/quize/static/images/buser.png')
#     # sql2 ="select jikwon_jik, jikwon_gen from jikwon"
#
#     jikwons = Jikwon.objects.all().values('jikwon_jik',"jikwon_gen")
#     df4 = pd.DataFrame.from_records(jikwons)
#     df4.columns=['직급','성별']
#     print(df4)
#     ctab= pd.crosstab(df4['직급'],df4["성별"])
#     # print(ctab)
#
#
#
#     return render(request, 'list.html', 
#                   {'list':df.to_html(index=False),
#                    'jik_group':df2.to_html(),
#                    'buser_group':df3.to_html(),
#                    'ctab':ctab.to_html(),
#
#
#                                          })
#
plt.rc('font', family='malgun gothic')

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')

def listFnc(request):
    # https://brownbears.tistory.com/101
    # 형식 : table1.objects.extra(tables=['table2'], where=['table2.id=group.num'])

    data = Jikwon.objects.extra(select = {'buser_name':'buser_name'}, 
                tables=['Buser'], 
                where=['Buser.buser_no=Jikwon.buser_num']).values\
                ('jikwon_no', 'jikwon_name', 'buser_name', 'jikwon_jik', 'jikwon_pay', 'jikwon_ibsail', 'jikwon_gen').\
                order_by('buser_name', 'jikwon_name')
               
    # values()로 부분추출 하지말고 필요한 요소만 dict에 담아서 추출
    pd.set_option('display.max_columns', 500) # 칼럼 모두 보기 옵션
    df = pd.DataFrame(data) 
    df.columns=['부서명', '사번', '직원명', '직급', '연봉', '입사', '성별'] 
    
    period = []
    for i in data.values('jikwon_ibsail'):
        period.append((date.today()).year - (i['jikwon_ibsail']).year)
    
    df['근무년수'] = period
    df.sort_values(by=['부서명','직원명'])
    del df['입사']
    # print(df)
   
    group = df['연봉'].groupby(df['부서명'])
    detail = {'sum':group.sum(), 'avg':group.mean()}
    
    group_j = df['연봉'].groupby(df['직급'])
    detail_j = {'sum':group_j.sum(), 'avg':group_j.mean()}
    
    result = group.agg(['sum','mean'])
    result.plot(kind='bar')
    plt.title("부서별 급여합, 급여평균")
    plt.ylabel("연봉")
    fig = plt.gcf()
    fig.savefig('django10_ex/quize/static/images/buser.png')
    
    ctab = pd.crosstab(df['성별'], df['직급'], margins=True)
    
    return render(request, 'show.html', 
                  {'df':df.to_html(), 
                   'detail':detail, 
                   'detail_j':detail_j, 
                   'ctab':ctab.to_html()})     
     
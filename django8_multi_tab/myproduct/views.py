from django.shortcuts import render
from myproduct.models import Maker, Product
import pandas as pd
import MySQLdb
import pickle
# Create your views here.
config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'maria123',
    'database':'productdb',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}

def Main(request):
    return render(request, 'main.html')

def List1(request):
    # 제조사 정보를 출력
    #1) Django ORM
    makers = Maker.objects.all()
    print(type(makers)) #.QuerySet   
    return render(request, 'list1.html', {'makers':makers})
    
    """
    # 2) Django ORM의 결과를 pd.DataFrame으로 저장 후 전송
    df = pd.DataFrame(list(Maker.objects.all().values()))
    # df = pd.DataFrame(list(Maker.objects.all().values('mname','tel')))
    # print(df)
    # return render(request, 'list1_1.html', {'makers':df.to_html(index=False)})

    #3) SQL문 사용   
    conn = MySQLdb.connect(**config) # ** dict 타입
    cursor = conn.cursor()
    sql = "select * from myproduct_maker"              
    cursor.execute(sql)
    rows = cursor.fetchall()
    print(type(rows)) # tuple
    # df = pd.DataFrame(rows)  
    # return render(request, 'list1_1.html', {'makers':df.to_html(index=False)})

    #4) SQL문 사용 - pandas 기능
    df = pd.read_sql("select * from myproduct_maker", conn)
    return render(request, 'list1_1.html', {'makers':df.to_html(index=False)})
    """
    
from django.db.models.aggregates import Count, Sum, Avg, StdDev, Variance
def List2(request):
    # 제품 정보를 출력
    #1) Django ORM
    products = Product.objects.all()
    pcount = len(products)
    
    
    # ORM 연습
    print(products.values_list()) # QuerySet
    print(products.aggregate(Count('price'))) # price 칼럼벨류의 갯수
    print(products.aggregate(Sum('price'))) # price 칼럼벨류의 합
    imsi = products.values('pname').annotate(Avg('price')) # pname: price벨류 평균
    print(imsi)
    for r in imsi:
        print(r)
    
    return render(request, 'list2.html', {'products':products, 'pcount':pcount})

def List3(request):
    mid = request.GET.get('id')
    products = Product.objects.filter(maker_name=mid) # where 조건
    
    pcount = len(products)
    return render(request, 'list2.html', {'products':products, 'pcount':pcount})


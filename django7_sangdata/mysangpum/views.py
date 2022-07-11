from django.shortcuts import render
import MySQLdb
from mysangpum.models import Sangdata
from django.http.response import HttpResponseRedirect
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

def MainFunc(request):
    return render(request, 'main.html')

def ListFunc(request):
    """
    sql = "select * from sangdata"
    conn = MySQLdb.connect(**config)
    cursor.execute(sql)
    datas = cursor.fetchall() #QuerySet이 아님 return type이 tuple
    """
    # 페이징 처리 X
    # datas = Sangdata.objects.all() # return type이 QuerySet   
    # return render(request, 'list.html',{'sangpums':datas})
    
    # 페이징 처리 O ----------
    
    datas = Sangdata.objects.all().order_by('-code')
    paginator = Paginator(datas, 5) # 페이지 당 5행 씩 출력
    
    try:
        page =request.GET.get('page')
    except:
        page = 1
        
    try:
        data = paginator.page(page)
        
    except PageNotAnInteger:     #page에 숫자가아닌 다른 값을 넣었을 떄        
        data =paginator.page(1)
        
    except EmptyPage:        
        data =paginator.page(paginator.num_pages())
        
    # 개별 페이지 표시용
    allpage = range(paginator.num_pages + 1) # 0부터 시작하니까 맞추기위해 +1
    
    return render(request, 'list2.html',{'sangpums':data, 'allpage':allpage})

def InsertFunc(request):
    
    return render(request, 'insert.html')

def InsertOkFunc(request):
    if request.method == "POST":
        # code = request.POST.get("code");
        # print(code)
        # 새로운 상품 code가 중복되는지 검사 후 insert 진행
        try:
            Sangdata.objects.get(code = request.POST.get("code")) #읽없는데 없다? except
            return render(request, 'insert.html',{'msg':'이미 등록된 번호입니다.'})
        except Exception as e:
            Sangdata(
                code = request.POST.get("code"),
                sang = request.POST.get("sang"),
                su = request.POST.get("su"),
                dan = request.POST.get("dan"),
                ).save()
                
        return HttpResponseRedirect('/sangpum/list')# 추가 후 목록보기
    

def UpdateFunc(request):
    data = Sangdata.objects.get(code=request.GET.get('code'))   
    return render(request, 'update.html',{'sang_one':data})

def UpdateOkFunc(request):
    if request.method == "POST":
        upRec = Sangdata.objects.get(code = request.POST.get("code"))
        upRec.code = request.POST.get("code")
        upRec.sang = request.POST.get("sang")
        upRec.su = request.POST.get("su")
        upRec.dan = request.POST.get("dan")
        upRec.save()
    
    return HttpResponseRedirect('/sangpum/list') # 수정 후 목록보기
def DeleteOkFunc(request):
    delRec = Sangdata.objects.get(code = request.GET.get("code"))
    delRec.delete()
    
    return HttpResponseRedirect('/sangpum/list') # 삭제 후 목록보기
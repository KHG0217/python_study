from django.shortcuts import render
from myguest.models import Guest
from datetime import datetime
from django.utils import timezone
from django.http.response import HttpResponseRedirect
from distributed.http.utils import redirect

# Create your views here.
def MainFunc(request):
    return render(request, 'main.html')

def SelectFunc(request):
    print(Guest.objects.filter(title__contains='안녕')) #title에 '안녕'이 있나? True False
    print(Guest.objects.get(id=1)) # 하나만 검색가능 (where 조건절이라고 생각)
    print(Guest.objects.filter(id=1))# 복수개 가능 (where 조건절이라고 생각)
    print(Guest.objects.filter(title='연습')) # '연습'이라고 title에 붙어있는거 다 가져옴    
    
    # 정렬
    gdata = Guest.objects.all()
    # gdata = Guest.objects.all().order_by('title') # 제목별 오름차순
    # gdata = Guest.objects.all().order_by('-title') # 제목별 내림차순
    # gdata = Guest.objects.all().order_by('-title', 'id') #제목별 내림차순 id 오름차순
    # gdata = Guest.objects.all().order_by('-id')[0:2] # id 내림차순 슬라이싱 가능
    
    return render(request, 'list.html', {'gdata':gdata})


def InsertFunc(request):
    
    return render(request, 'insert.html')

def InsertOkFunc(request):
    if request.method =="POST":
        # print(request.POST.get('title'))
        Guest(
        #    칼럼명=    넘어오는 form의 name값
            title = request.POST.get('title'),
            content = request.POST['content'],
            # regdate = datetime.now() 
            regdate = timezone.now()                 
            ).save() #Guest
    
    return HttpResponseRedirect('/guest/select') # 추가후 목록보기. Redirect 방식   
    # return redirect('/guest/select')
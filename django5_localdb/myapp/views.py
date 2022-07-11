from django.shortcuts import render
from myapp.models import Article

# Create your views here.
def main(request):
    return render(request, 'main.html')

def showdb(request):
    # 1. SQL문을 직접 사용해서 html에 전달
    # 2. Django의 orm 기능을 사용하는 방법 (권장)
    
    datas = Article.objects.all() #select *from Article 
    print(datas,type(datas))    # QuerySet 
    print(datas[0].name) # 마스크 
    
    return render(request, 'list.html', {'articles':datas})

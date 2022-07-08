from django.shortcuts import render
from django.views.generic.base import TemplateView

# Create your views here.
def MainFunc(request):
    return render(request,'index.html')

class MyCallView(TemplateView): # 기능을 확장해서 쓸 수있는 장점이 있다.
    template_name ='callget.html'

def InsertFunc(request):
    # return render(request,'insert.html')
    
    # 같은 요청명에 대해 get, post를 구분해서 처리 가능
    if request.method == 'GET':
        return render(request,'insert.html')
    elif request.method == 'POST':
        buser = request.POST.get('buser')
        irum = request.POST['irum']
        print(buser, irum)
        
         # buser,irum 으로 뭔가를 하면 된다.
        msg1 = '부서명 :' + buser
        msg2 = '직원이름 :' + irum
        context ={'msg1':msg1,'msg2':msg2} #dict 타입
        return render(request, 'show.html',context)
    else:
        print('요청 오류')    

# def InsertFuncOk(request):
#     # buser = request.GET.get('buser') # get 방식으로 데이터를 받아오는법
#     # irum = request.GET['irum'] ## get 방식으로 데이터를 받아오는법 2
#     buser = request.POST.get('buser')
#     irum = request.POST['irum']
#     # buser,irum 으로 뭔가를 하면 된다.
#     msg1 = '부서명 :' + buser
#     msg2 = '직원이름 :' + irum
#     context ={'msg1':msg1,'msg2':msg2}
#     return render(request, 'show.html',context)
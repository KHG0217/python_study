from django.shortcuts import render

# Create your views here.
#            키워드임
def mainFunc(request):
    name = "홍길동"
    return render(request, 'main.html', {'msg':name,'test':'이것도가나'}) #templates에 main.html을 찾아감 forworading 방식 ex)''<-경로를 안줬으니 ->main.html으로 가!
#                                                        데이터 - dict타입으로 데이터 넘기기

def helloFunc(request):
    str = "<h1>출력을 위한 작업 </h1>"
    return render(request, 'disp.html',{'key':str})
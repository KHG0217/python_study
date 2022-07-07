from django.shortcuts import render

# Create your views here.
#            키워드임
def mainFunc(request):
    name = "홍길동"
    return render(request, 'main.html', {'msg':name}) #templates에 main.html을 찾아감 forworading 방식 ex)''<-경로를 안줬으니 ->main.html으로 가!
#                                                        데이터 - dict타입으로 데이터 넘기기
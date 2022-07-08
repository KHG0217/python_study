from django.shortcuts import render
from django.http.response import HttpResponseRedirect

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')

def setOsFunc(request):
    # print(request.GET) # <QueryDict: {}>
    if "favorite_os" in request.GET: #  "favorite_os"에 (request.GET) 담겨진 값이 있다면(현재 경로에서 담는거임)
        print(request.GET['favorite_os'])
        # request.session['세션키']
        request.session['f_os'] = request.GET['favorite_os'] # 세션 생성 f_os라는 킷값으로 favorite_os의 벨류를 담아 세션을 만든다.
        return HttpResponseRedirect("/showos") # redirect 방식 (클라이언트를 통해서 서버에 요청을 한다.)
        # urls에 path('showos',views.showOsFunc), 를 만나게 하기 위해
    
    else:
        return render(request, 'selectos.html') #forward 방식 :서버에서 서버파일을 호출 

def showOsFunc(request):
    
    dict_context = {} #session 자료를 html 파일에 전달할 목적으로 생성
    
    if "f_os" in request.session:   #request 세션값 중에 있니?
        print('유효시간 :', request.session.get_expiry_age())
        dict_context['sel_os'] = request.session['f_os']
        dict_context['message'] = "당신이 선택한 운영체제는 %s" %request.session['f_os'] # 기본 유효시간 30분
    else:
        dict_context['sel_os'] = None
        dict_context['message'] ="운영체제를 선택하지 않았네요"
        
    #참고 : 특전 세션 삭제 request.session['key']
    # set_expory(0) 하면 브라우저가 닫힐 떄 세션이 해제됨
    
    request.session.set_expiry(5) #5초 동안 세션이 유효. 기본값은 30분
    return render(request, 'show.html',dict_context) #forward 방식
        
        
        

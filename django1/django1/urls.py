"""
    클라이언트의 모든 요청을 받는 곳/full stack framwork (프론트 백엔드를 모든걸 맡아서 함) =java는 web.xml
django1 URL Configuration
    
The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views
#url주소 만들기
urlpatterns = [
    path('admin/', admin.site.urls),
    
    path('', views.mainFunc), #apprication 마다 view가 있으니 쓸려는 view 패키지명 잘 확인/ 경로에 ''<-아무것도 안했을때 ,갈 view forwarding 방식
    path('hello', views.helloFunc), 
    
]

from django.urls import path
from gpapp import views

urlpatterns = [
    path('insert',views.InsertFunc), #GET 방식
    # path('insertok',views.InsertFuncOk), 
]
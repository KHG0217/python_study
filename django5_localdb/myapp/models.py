from django.db import models

# Create your models here.
class Article(models.Model):    #table명 Aricle
    code = models.CharField(max_length=10)  #varchr(10)
    # models.TextField #많은양의 문자쓸때
    name = models.CharField(max_length=20)  #varchr(20)
    price = models.IntegerField()   # number
    pub_date = models.DateTimeField()  # date
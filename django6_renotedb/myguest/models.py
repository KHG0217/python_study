from django.db import models

# Create your models here.
class Guest(models.Model):
    # myno = models.AutoField(auto_created = True, primary_key = True) #자동으로 만들어지는 id 대신 myno가 생성됨
    title = models.CharField(max_length=50)
    content = models.TextField()
    regdate = models.DateTimeField()
    
    # 정렬하기 방법 2
    class Meta:     #subClass 느낌
        # ordering = ('title', 'id')   #튜플타입으로 줘야함
        ordering = ('-id',)
    

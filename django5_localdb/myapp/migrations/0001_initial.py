# Generated by Django 4.0.6 on 2022-07-11 02:16

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Article',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),   #django에서 id를 자동으로 생성해줌
                ('code', models.CharField(max_length=10)),
                ('name', models.CharField(max_length=20)),
                ('price', models.IntegerField()),
                ('pub_date', models.DateTimeField()),
            ],
        ),
    ]

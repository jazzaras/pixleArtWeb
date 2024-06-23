
from django.contrib import admin
from django.urls import path, include
from . import views

app_name = 'pixle'
urlpatterns = [
    path('', views.home, name="home"),
    # path('converted', views.conv, name="conv"),
]


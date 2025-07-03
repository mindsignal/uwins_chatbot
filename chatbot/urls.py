from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from . import views

urlpatterns = [
    path('', views.home,name="home"),
    #path('chattrain', views.chattrain, name='chattrain'),
    path('chatanswer_hubok', views.chatanswer_hubok, name='chatanswer_hubok'),
    path('chatanswer_delok', views.chatanswer_delok, name='chatanswer_delok'),
    path('chatanswer_hubok_correct', views.chatanswer_hubok_correct, name='chatanswer_hubok_correct'),
    path('chatanswer_delok_correct', views.chatanswer_delok_correct, name='chatanswer_delok_correct'),
    path('chatanswer_hubok_incorrect', views.chatanswer_hubok_incorrect, name='chatanswer_hubok_incorrect'),
    path('chatanswer_delok_incorrect', views.chatanswer_delok_incorrect, name='chatanswer_delok_incorrect'),
    path('delok',views.delok, name="delok"),
    path('hubok', views.hubok, name="hubok"),
    path('sugang', views.sugang, name="sugang"),
    path('chatanswer_sugang', views.chatanswer_sugang,name='chatanswer_sugang'),
    path('welfare', views.welfare, name="welfare"),
    path('chatanswer_welfare', views.chatanswer_welfare, name='chatanswer_welfare'),
    path('unknown_question', views.unknown_question, name='unknown_question')

]

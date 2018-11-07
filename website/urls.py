from django.urls import path
from . import views

app_name = "website"

urlpatterns = [
    path('', views.index),
    path('camera', views.camera),
    path('uploadImage', views.uploadImage),
    path('chatbot', views.chatbot),
    path('webhook', views.webhook),
    path('send_message', views.send_message),
]
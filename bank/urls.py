from django.urls import path
from . import views

app_name = "bank"

urlpatterns = [
    path('login', views.login),
    path('main', views.main),
    path('transfer', views.transfer),
]
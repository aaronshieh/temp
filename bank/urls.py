from django.urls import path
from . import views

app_name = "bank"

urlpatterns = [
    path('create', views.create_account),
    path('login', views.login),
    path('main', views.main),
    path('transfer', views.transfer),
]
from django.urls import path
from . import views

app_name = "face_utils"

urlpatterns = [
    path('identify_face', views.identify_face),
]
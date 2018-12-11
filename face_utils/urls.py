from django.urls import path
from . import views

app_name = "face_utils"

urlpatterns = [
    path('identify_face', views.identify_face),
    path('identify_emotion', views.identify_emotion),
    path('new_member', views.new_member),
    path('recognize_face', views.recognize_face),
    path('get_members', views.get_members),
    path('del_members', views.del_members),
]
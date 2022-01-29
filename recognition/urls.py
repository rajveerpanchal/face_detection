from django.urls import path, include
from recognition import views


urlpatterns = [
    path('', views.index, name='index'),
    path('face_detect', views.facecam_feed, name='face_detect'),
    ]
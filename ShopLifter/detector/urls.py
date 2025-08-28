# detector/urls.py
from django.urls import path
from .views import home, predict, upload_page

urlpatterns = [
    # path("", home, name="home"),
    path("predict/", predict, name="predict"),
    path("", upload_page, name="upload"),
]

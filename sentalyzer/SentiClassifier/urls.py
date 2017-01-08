from django.conf.urls import url
from SentiClassifier.views import *
from . import views

urlpatterns = [
    url(r'^index/', homePage.as_view()),
    url(r'^result/', resultView.as_view()),
]
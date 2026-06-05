# classifier/urls.py

from django.urls import path
from .views import classify_message

urlpatterns = [
    path("classify/", classify_message, name="classify_message")
]
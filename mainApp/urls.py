from django.urls import path
from .views import Home, PredictionPage

urlpatterns = [
    path('', Home, name='homePage'),
    path('Covid-19-Test/', PredictionPage, name='predictionPage'),
]

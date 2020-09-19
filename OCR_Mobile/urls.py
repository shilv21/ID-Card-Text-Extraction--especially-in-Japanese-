from django.urls import path, include
import OCR_Mobile.views
urlpatterns = [
    path('', OCR_Mobile.views.simple_upload, name='TestRun'),
    path('checking/',OCR_Mobile.views.checking, name = 'Checking')
]

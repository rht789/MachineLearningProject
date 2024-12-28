from django.urls import path
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'analysis', views.MLAnalysisViewSet, basename='analysis')

app_name = 'ml_app'

urlpatterns = router.urls

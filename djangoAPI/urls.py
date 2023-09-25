
from django.contrib import admin
from django.urls import path, include
from systemsolver import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/matrix/triangularize/', views.triangularize_matrix, name='triangularize-matrix'),
    path('api/matrix/elimination_gaussiana/', views.elimination_gaussiana, name='elimination_gaussiana'),

]




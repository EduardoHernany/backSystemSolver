
from django.contrib import admin
from django.urls import path, include
from systemsolver import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/matrix/elimination_gaussiana/', views.jordan_matrix_matrix, name='triangularize-matrix'),
    #path('api/matrix/elimination_gaussiana/', views.elimination_gaussiana, name='elimination_gaussiana'),

    path('api/matrix/elimination_jordan/', views.jordan_matrix, name='elimination_jordan'),
    path('api/matrix/elimination_LU/', views.lu_matrix, name='elimination_LU'),
    path('api/matrix/elimination_jacobi/', views.jacobi_matrix, name='elimination_jacobi'),
    path('api/matrix/elimination_newton/', views.newton_matrix, name='elimination_newton'),

]




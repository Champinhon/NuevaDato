from django.contrib import admin
from django.urls import path
from account import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name="index"),
    path('registro/',views.registro_view,name='registro'),
    path('logout/',views.logout_view,name='logout'),
    path('login/',views.login_view,name='login'),
    path('dashboard/',views.dashboard_view,name='dashboard'),
    path('upload/', views.upload_file, name='upload_file'),

]

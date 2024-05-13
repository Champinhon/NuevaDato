# from django.conf.urls import url
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

#-------------------------General(Dashboards,Widgets & Layout)---------------------------------------
    path('', views.index, name='index'),
    path('upload/', views.upload_file, name='upload_file'),
    path('get_latest_chats/', views.get_latest_chats, name='get_latest_chats'),
    path('signup_home',views.signup_home,name='signup_home'),
    path('login_home', views.login_home, name="login_home"),
    path('logout_view', views.logout_view, name="logout_view"),
    #path('stream_response', views.stream_response,  name="stream_response"),
    #path('plan_economico', views.plan_economico,  name="plan_economico"),
    #path('vista_generar_texto_incremental', views.vista_generar_texto_incremental, name="vista_generar_texto_incremental"),
]
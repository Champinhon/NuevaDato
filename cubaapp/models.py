from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    message = models.TextField(null=True)
    response = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'
class Archivo(models.Model):
    archivo = models.FileField(blank=True, null=True)

    def __str__(self):
        return str(self.archivo)

    def get_archivo_url(self):
        return self.archivo.url
class Imagen(models.Model):
    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
    url = models.URLField()
    fecha_creacion = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Imagen de {self.usuario.username} - {self.fecha_creacion}"
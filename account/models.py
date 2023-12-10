from django.db import models

class Archivo(models.Model):
    archivo = models.FileField(blank=True, null=True)

    def __str__(self):
        return str(self.archivo)

    def get_archivo_url(self):
        return self.archivo.url

from django import forms
from django.forms import ModelForm

from .models import *


class ArchivoForm(forms.ModelForm):
    class Meta:
        model = Archivo
        fields = ['archivo']
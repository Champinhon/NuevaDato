# Generated by Django 4.2.4 on 2023-12-09 16:41

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("account", "0011_remove_opcion_pregunta_remove_pregunta_examen_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="Archivo",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "tipo_archivo",
                    models.CharField(
                        choices=[
                            ("excel", "Archivo Excel"),
                            ("csv", "Archivo CSV"),
                            ("pdf", "Archivo PDF"),
                        ],
                        max_length=10,
                    ),
                ),
                ("archivo", models.FileField(upload_to="archivos/")),
            ],
        ),
    ]

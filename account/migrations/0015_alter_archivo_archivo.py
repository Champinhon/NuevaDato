# Generated by Django 4.2.4 on 2023-12-09 17:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("account", "0014_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="archivo",
            name="archivo",
            field=models.FileField(blank=True, null=True, upload_to=""),
        ),
    ]

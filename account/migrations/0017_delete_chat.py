# Generated by Django 4.2.4 on 2023-12-10 02:06

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("account", "0016_chat"),
    ]

    operations = [
        migrations.DeleteModel(
            name="Chat",
        ),
    ]

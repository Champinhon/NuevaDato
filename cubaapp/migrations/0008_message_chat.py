# Generated by Django 4.1.4 on 2024-01-09 14:05

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("cubaapp", "0007_rename_sender_message_user_remove_message_chat"),
    ]

    operations = [
        migrations.AddField(
            model_name="message",
            name="chat",
            field=models.ForeignKey(
                default=1,
                on_delete=django.db.models.deletion.CASCADE,
                to="cubaapp.chat",
            ),
        ),
    ]
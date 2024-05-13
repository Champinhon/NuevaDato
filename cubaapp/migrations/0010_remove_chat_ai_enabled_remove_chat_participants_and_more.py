# Generated by Django 4.1.4 on 2024-01-09 17:22

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("cubaapp", "0009_delete_task_chat_ai_enabled_message_is_ai_message_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="chat",
            name="ai_enabled",
        ),
        migrations.RemoveField(
            model_name="chat",
            name="participants",
        ),
        migrations.AddField(
            model_name="chat",
            name="message",
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name="chat",
            name="response",
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name="chat",
            name="user",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="chat",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.DeleteModel(
            name="Message",
        ),
    ]

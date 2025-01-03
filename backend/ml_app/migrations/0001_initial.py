# Generated by Django 5.1.4 on 2024-12-22 17:46

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="UploadedDataset",
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
                ("file", models.FileField(upload_to="datasets/")),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                ("original_filename", models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name="AnalysisResult",
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
                ("objective", models.CharField(max_length=50)),
                ("target_column", models.CharField(blank=True, max_length=100)),
                ("metrics", models.JSONField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "dataset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="analysis_results",
                        to="ml_app.uploadeddataset",
                    ),
                ),
            ],
        ),
    ]

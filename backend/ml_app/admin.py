from django.contrib import admin
from .models import UploadedDataset, AnalysisResult

@admin.register(UploadedDataset)
class UploadedDatasetAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'uploaded_at')
    search_fields = ('original_filename',)

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'objective', 'created_at')
    list_filter = ('objective', 'created_at')
    search_fields = ('dataset__original_filename',)

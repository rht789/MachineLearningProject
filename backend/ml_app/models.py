from django.db import models
from django.utils import timezone

class UploadedDataset(models.Model):
    file = models.FileField(upload_to='uploads/')
    columns = models.JSONField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    original_filename = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)  # To track if file is still available
    
    class Meta:
        ordering = ['-uploaded_at']  # Most recent first
    
    def __str__(self):
        return f"{self.original_filename} (uploaded {self.uploaded_at.strftime('%Y-%m-%d')})"

    def get_file_info(self):
        return {
            'id': self.id,
            'filename': self.original_filename,
            'uploaded_at': self.uploaded_at.strftime('%Y-%m-%d %H:%M'),
            'columns': self.columns
        }


class AnalysisResult(models.Model):
    dataset = models.ForeignKey(
        UploadedDataset,
        on_delete=models.CASCADE,
        related_name='analysis_results'
    )
    objective = models.CharField(max_length=50)
    target_column = models.CharField(max_length=100, blank=True)
    metrics = models.JSONField()  # Will store results for all algorithms
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis for {self.dataset} ({self.objective})"

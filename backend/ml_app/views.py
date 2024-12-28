from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import UploadedDataset, AnalysisResult
from django.shortcuts import get_object_or_404
from .ml_utils import analyze_dataset
import pandas as pd
from .ml_utils import MLModel
import logging
import json
import numpy as np
from django.core.serializers.json import DjangoJSONEncoder
from django.http import HttpResponse
from .report_utils import generate_html_report, generate_pdf_report, debug_wkhtmltopdf
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class NumpyEncoder(DjangoJSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class MLAnalysisViewSet(viewsets.ViewSet):
    @action(detail=False, methods=['POST'])
    def upload_file(self, request):
        try:
            file = request.FILES.get('file')
            if not file:
                return Response(
                    {'error': 'File is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate file type
            if not file.name.endswith('.csv'):
                return Response(
                    {'error': 'Only CSV files are supported'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                # Read CSV with pandas, handling different encodings and separators
                try:
                    df = pd.read_csv(file)
                except UnicodeDecodeError:
                    # Try different encoding if UTF-8 fails
                    df = pd.read_csv(file, encoding='latin1')
                except:
                    # Try with different separator if comma fails
                    df = pd.read_csv(file, sep=';')

                # Clean and prepare column names
                columns = [str(col).strip() for col in df.columns]
                
                # Remove any empty or invalid column names
                columns = [col for col in columns if col and not col.isspace()]

                # Save the file
                dataset = UploadedDataset.objects.create(
                    file=file,
                    original_filename=file.name,
                    columns=columns
                )

                return Response({
                    'file_id': dataset.id,
                    'columns': columns,
                    'filename': file.name,
                    'row_count': len(df)
                })

            except Exception as e:
                return Response(
                    {'error': f'Error reading CSV file: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

        except Exception as e:
            print(f"Error processing file: {str(e)}")  # For debugging
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def create(self, request):
        try:
            file_id = request.data.get('file_id')
            objective = request.data.get('objective')
            target_column = request.data.get('target_column')

            if not all([file_id, objective]):
                return Response(
                    {'error': 'File ID and objective are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get the dataset
            dataset = get_object_or_404(UploadedDataset, id=file_id)

            # Validate target column if needed
            if objective in ['regression', 'classification']:
                if not target_column:
                    return Response(
                        {'error': 'Target column is required for regression and classification'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

            # Perform ML analysis
            analysis_results = analyze_dataset(
                dataset.file.path,
                objective,
                target_column=target_column
            )

            # Save results with custom JSON encoder
            result = AnalysisResult.objects.create(
                dataset=dataset,
                objective=objective,
                target_column=target_column or '',
                metrics=json.loads(json.dumps(analysis_results, cls=NumpyEncoder))
            )

            logger.info(f"Created analysis result with ID: {result.id}")
            response_data = {
                'message': 'Analysis completed successfully',
                'id': result.id,
                'results': json.loads(json.dumps(analysis_results, cls=NumpyEncoder))
            }
            logger.info(f"Sending response with analysis ID: {result.id}")
            return Response(response_data)
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return Response(
                {'error': f'Analysis failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['GET'])
    def previous_analyses(self, request):
        analyses = AnalysisResult.objects.all().order_by('-created_at')[:5]
        results = [{
            'id': analysis.id,
            'dataset': analysis.dataset.original_filename,
            'objective': analysis.objective,
            'metrics': analysis.metrics,
            'created_at': analysis.created_at
        } for analysis in analyses]
        
        return Response(results)

    @action(detail=False, methods=['GET'])
    def previous_uploads(self, request):
        try:
            datasets = UploadedDataset.objects.filter(is_active=True).order_by('-uploaded_at')[:10]  # Get last 10 uploads
            return Response([
                dataset.get_file_info() for dataset in datasets
            ])
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['GET'])
    def available_algorithms(self, request):
        """Return available algorithms for each objective"""
        return Response({
            'regression': list(MLModel.ALGORITHMS['regression'].keys()),
            'classification': list(MLModel.ALGORITHMS['classification'].keys()),
            'clustering': list(MLModel.ALGORITHMS['clustering'].keys())
        })

    @action(detail=False, methods=['GET'])
    def check_pdf_setup(self, request):
        """Debug endpoint to check PDF generation setup"""
        try:
            debug_info = debug_wkhtmltopdf()
            return Response(debug_info)
        except Exception as e:
            logger.error(f"PDF setup check failed: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['GET'])
    def download_pdf(self, request, pk=None):
        """Download analysis results as PDF"""
        try:
            analysis = get_object_or_404(AnalysisResult, id=pk)
            logger.info(f"Generating PDF report for analysis {pk}")
            
            try:
                pdf_content = generate_pdf_report(analysis)
                
                if not pdf_content:
                    raise Exception("Generated PDF content is empty")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"analysis_report_{pk}_{timestamp}.pdf"
                
                # Set response headers for PDF
                response = HttpResponse(
                    content=pdf_content,
                    content_type='application/pdf'
                )
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                response['Content-Length'] = len(pdf_content)
                response['Accept-Ranges'] = 'bytes'
                response['Access-Control-Expose-Headers'] = 'Content-Disposition'
                response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response['Pragma'] = 'no-cache'
                response['Expires'] = '0'
                
                return response
                
            except Exception as e:
                logger.error(f"PDF generation failed: {str(e)}")
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
        except Exception as e:
            logger.error(f"PDF download failed: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['GET'])
    def download_html(self, request, pk=None):
        """Download analysis results as HTML"""
        try:
            analysis = get_object_or_404(AnalysisResult, id=pk)
            logger.info(f"Generating HTML report for analysis {pk}")
            
            try:
                html_content = generate_html_report(analysis)
                
                if not html_content:
                    raise Exception("Generated HTML content is empty")
                
                response = HttpResponse(html_content, content_type='text/html')
                response['Content-Disposition'] = f'attachment; filename="analysis_report_{pk}.html"'
                return response
                
            except Exception as e:
                logger.error(f"HTML generation failed: {str(e)}")
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
        except Exception as e:
            logger.error(f"HTML download failed: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

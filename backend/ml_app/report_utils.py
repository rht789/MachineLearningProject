from django.template.loader import render_to_string
import json
from datetime import datetime
import os
import platform
import subprocess
import tempfile
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def generate_html_report(analysis_result):
    """Generate HTML report from analysis results"""
    try:
        # Convert numpy values to Python native types
        metrics = json.loads(json.dumps(analysis_result.metrics))
        
        context = {
            'analysis_date': analysis_result.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'objective': analysis_result.objective.title(),
            'target_column': analysis_result.target_column,
            'metrics': metrics,
            'dataset_name': analysis_result.dataset.original_filename
        }
        
        html_content = render_to_string('ml_app/report_template.html', context)
        return html_content
        
    except Exception as e:
        logger.error(f"HTML Generation Error: {str(e)}")
        raise Exception(f"Failed to generate HTML report: {str(e)}")

def generate_pdf_report(analysis_result):
    """Generate PDF report from analysis results"""
    try:
        # Generate HTML content first
        html_content = generate_html_report(analysis_result)
        logger.info("HTML content generated successfully")

        # Create temporary files with unique names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_dir = tempfile.gettempdir()
        temp_html_path = os.path.join(temp_dir, f'report_{analysis_result.id}_{timestamp}.html')
        temp_pdf_path = os.path.join(temp_dir, f'report_{analysis_result.id}_{timestamp}.pdf')
        
        logger.info(f"Using temporary files: HTML={temp_html_path}, PDF={temp_pdf_path}")

        # Add base tag to HTML for proper resource loading
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
            <base href="file:///">
            <style>
                @page {{
                    margin: 2cm;
                    size: A4;
                }}
                body {{
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Write HTML content to file
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Path to wkhtmltopdf
        wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        
        if not os.path.exists(wkhtmltopdf_path):
            raise Exception(f"wkhtmltopdf not found at {wkhtmltopdf_path}")

        try:
            # Command to generate PDF with additional options
            cmd = [
                wkhtmltopdf_path,
                '--quiet',
                '--page-size', 'A4',
                '--encoding', 'utf-8',
                '--margin-top', '20',
                '--margin-right', '20',
                '--margin-bottom', '20',
                '--margin-left', '20',
                '--enable-local-file-access',
                '--disable-smart-shrinking',
                '--print-media-type',
                '--no-background',
                '--javascript-delay', '1000',
                temp_html_path,
                temp_pdf_path
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run wkhtmltopdf with a timeout
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30  # 30 seconds timeout
            )
            
            # Check if PDF was created
            if not os.path.exists(temp_pdf_path):
                raise Exception("PDF file was not created")
                
            # Check PDF file size
            pdf_size = os.path.getsize(temp_pdf_path)
            if pdf_size == 0:
                raise Exception("Generated PDF file is empty")
                
            logger.info(f"PDF generated successfully, size: {pdf_size} bytes")
            
            # Read the PDF content
            with open(temp_pdf_path, 'rb') as f:
                pdf_content = f.read()
                
            return pdf_content

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout
            logger.error(f"PDF Generation Error: {error_msg}")
            raise Exception(f"wkhtmltopdf failed: {error_msg}")
            
        except subprocess.TimeoutExpired:
            logger.error("PDF generation timed out")
            raise Exception("PDF generation timed out after 30 seconds")
            
        except Exception as e:
            logger.error(f"PDF Generation Error: {str(e)}")
            raise Exception(f"Failed to generate PDF: {str(e)}")
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)
                if os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {str(e)}")

    except Exception as e:
        logger.error(f"Overall PDF Generation Error: {str(e)}")
        raise Exception(f"Failed to generate PDF report: {str(e)}")

def debug_wkhtmltopdf():
    """Debug function to check wkhtmltopdf installation"""
    try:
        wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        
        debug_info = {
            'wkhtmltopdf_path': wkhtmltopdf_path,
            'exists': os.path.exists(wkhtmltopdf_path),
            'version': None,
            'test_result': None,
            'temp_dir': tempfile.gettempdir(),
            'temp_dir_writable': os.access(tempfile.gettempdir(), os.W_OK)
        }
        
        if debug_info['exists']:
            try:
                # Get version
                result = subprocess.run([wkhtmltopdf_path, '-V'], 
                                     capture_output=True, 
                                     text=True)
                debug_info['version'] = result.stdout.strip()
                
                # Test PDF generation
                test_html = "<html><body><h1>Test PDF Generation</h1></body></html>"
                temp_html = os.path.join(tempfile.gettempdir(), 'test.html')
                temp_pdf = os.path.join(tempfile.gettempdir(), 'test.pdf')
                
                with open(temp_html, 'w', encoding='utf-8') as f:
                    f.write(test_html)
                
                test_cmd = [
                    wkhtmltopdf_path,
                    '--quiet',
                    temp_html,
                    temp_pdf
                ]
                
                test_result = subprocess.run(test_cmd, 
                                          capture_output=True, 
                                          text=True)
                
                if test_result.returncode == 0 and os.path.exists(temp_pdf):
                    debug_info['test_result'] = 'Success'
                else:
                    debug_info['test_result'] = f'Failed: {test_result.stderr}'
                
                # Clean up
                if os.path.exists(temp_html):
                    os.unlink(temp_html)
                if os.path.exists(temp_pdf):
                    os.unlink(temp_pdf)
                    
            except Exception as e:
                debug_info['test_result'] = f'Error: {str(e)}'
        
        return debug_info
        
    except Exception as e:
        return {'error': str(e)} 
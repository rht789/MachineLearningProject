// src/components/ResultsDisplay.js
import React from 'react';
import './ResultsDisplay.css';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell } from 'recharts';
import { 
    CheckCircle2, 
    XCircle, 
    Database, 
    Layers, 
    Scale, 
    Target, 
    FileStack,
    AlertCircle,
    Table,
    Rows,
    Columns,
    LineChart,
    Calculator,
    Ruler,
    BarChart2, 
    Percent,
    Activity,
    FileDown,
    FileText,
    Loader,
    AlertTriangle,
    ChevronDown
} from 'lucide-react';

function ResultsDisplay({ results, objective, analysisId }) {
    const [downloadState, setDownloadState] = React.useState({
        loading: false,
        error: null
    });
    const [showClusteringMetrics, setShowClusteringMetrics] = React.useState(false);

    const handleDownload = async (format) => {
        if (!analysisId) {
            setDownloadState({ 
                loading: false, 
                error: 'Analysis ID is missing. Please try running the analysis again.' 
            });
            return;
        }

        try {
            setDownloadState({ loading: true, error: null });
            
            const response = await fetch(`http://localhost:8000/api/analysis/${analysisId}/download_${format}/`, {
                method: 'GET',
                headers: {
                    'Accept': format === 'html' ? 'text/html' : 'application/pdf',
                },
                credentials: 'include'
            });

            if (!response.ok) {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `Failed to download ${format.toUpperCase()}`);
                } else {
                    throw new Error(`Failed to download ${format.toUpperCase()}: ${response.statusText}`);
                }
            }

            // For binary files (PDF), use blob(). For HTML, use text()
            const content = format === 'html' ? await response.text() : await response.blob();
            
            if ((format === 'pdf' && content.size === 0) || (format === 'html' && !content)) {
                throw new Error(`Downloaded ${format.toUpperCase()} is empty`);
            }

            // Create download link
            const url = format === 'html' ? 
                URL.createObjectURL(new Blob([content], { type: 'text/html' })) : 
                URL.createObjectURL(content);

            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `analysis_report_${analysisId}.${format}`;
            document.body.appendChild(a);
            a.click();
            
            // Cleanup
            setTimeout(() => {
                URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);
            
            setDownloadState({ loading: false, error: null });
        } catch (error) {
            // Only log and show errors that aren't related to ad blockers
            if (!error.message.includes('ERR_BLOCKED_BY_CLIENT')) {
                console.error(`Error downloading ${format}:`, error);
                setDownloadState({ 
                    loading: false, 
                    error: error.message || `Failed to download ${format.toUpperCase()}` 
                });
            } else {
                // If it's blocked by client but download worked, reset state
                setDownloadState({ loading: false, error: null });
            }
        }
    };

    if (!results || !results.algorithm_results) return null;

    const { algorithm_results, data_shape, preprocessing_info } = results;

    // Function to format metrics for display
    const formatMetric = (value) => {
        if (typeof value === 'object') return null; // Skip objects
        return typeof value === 'number' ? value.toFixed(4) : value;
    };

    // Function to get color based on metric value
    const getMetricColor = (value) => {
        if (typeof value !== 'number') return 'inherit';
        if (value > 0.9) return '#4CAF50';  // Green for excellent
        if (value > 0.7) return '#FFA726';  // Orange for good
        return '#EF5350';  // Red for needs improvement
    };

    // Prepare chart data
    const prepareChartData = () => {
        const chartData = [];
        Object.entries(algorithm_results).forEach(([algorithm, metrics]) => {
            // Skip non-metric data
            if (algorithm === 'data_distribution' || 
                algorithm === 'balancing_info' || 
                algorithm === 'target_classes' || 
                algorithm === 'class_distribution') {
                return;
            }

            // Format algorithm name
            const formattedName = algorithm.split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');

            // Add metrics to chart data
            chartData.push({
                name: formattedName,
                ...(objective === 'classification' ? {
                    accuracy: metrics.accuracy,
                    precision: metrics.precision,
                    recall: metrics.recall,
                    f1_score: metrics.f1_score
                } : {
                    r2_score: metrics.r2_score,
                    mse: metrics.mse,
                    rmse: metrics.rmse
                })
            });
        });
        return chartData;
    };

    // Filter metrics to show only numeric values
    const filterMetrics = (metrics) => {
        const filtered = {};
        Object.entries(metrics).forEach(([key, value]) => {
            if (typeof value === 'number') {
                filtered[key] = value;
            }
        });
        return filtered;
    };

    // Function to get icon for preprocessing item
    const getPreprocessingIcon = (key) => {
        switch(key) {
            case 'missing_values_handled':
                return <Database size={18} />;
            case 'categorical_columns':
                return <Layers size={18} />;
            case 'scaled_features':
                return <Scale size={18} />;
            case 'target_type':
                return <Target size={18} />;
            case 'small_dataset_handling':
                return <FileStack size={18} />;
            default:
                return <AlertCircle size={18} />;
        }
    };

    // Function to render value with appropriate icon
    const renderValueWithIcon = (value, valueClass) => {
        if (typeof value === 'boolean') {
            return value ? (
                <span className={`preprocessing-value ${valueClass}`}>
                    <CheckCircle2 size={18} className="value-icon" />
                    Yes
                </span>
            ) : (
                <span className={`preprocessing-value ${valueClass}`}>
                    <XCircle size={18} className="value-icon" />
                    No
                </span>
            );
        }
        return (
            <span className={`preprocessing-value ${valueClass}`}>
                {value}
            </span>
        );
    };

    // Update the metric icon selection
    const getMetricIcon = (metric) => {
        switch(metric.toLowerCase()) {
            case 'accuracy':
                return <Percent size={16} />;
            case 'precision':
                return <Target size={16} />;
            case 'recall':
                return <Activity size={16} />;
            case 'f1_score':
                return <BarChart2 size={16} />;
            case 'r2_score':
                return <Calculator size={16} />;
            case 'mse':
            case 'rmse':
                return <AlertCircle size={16} />;
            default:
                return null;
        }
    };

    // Function to get metric icon for clustering
    const getClusteringMetricIcon = (metric) => {
        switch(metric.toLowerCase()) {
            case 'silhouette_score':
                return <Activity size={16} />;  // Shows cluster separation quality
            case 'n_clusters':
                return <Layers size={16} />;    // Shows number of clusters
            case 'inertia':
                return <Target size={16} />;    // Shows cluster compactness
            default:
                return null;
        }
    };

    // Function to get color based on clustering metric
    const getClusteringMetricColor = (metric, value) => {
        if (typeof value !== 'number') return 'inherit';
        switch(metric.toLowerCase()) {
            case 'silhouette_score':
                // Silhouette score ranges from -1 to 1, where higher is better
                if (value > 0.7) return '#4CAF50';  // Good
                if (value > 0.5) return '#FFA726';  // Moderate
                return '#EF5350';                   // Needs improvement
            case 'inertia':
                // Lower inertia is better, but scale depends on data
                return '#2196F3';  // Use neutral color
            case 'n_clusters':
                return '#9C27B0';  // Use neutral color for number of clusters
            default:
                return 'inherit';
        }
    };

    // Prepare chart data for clustering
    const prepareClusteringChartData = () => {
        const chartData = [];
        Object.entries(algorithm_results).forEach(([algorithm, metrics]) => {
            // Skip non-metric data
            if (algorithm === 'data_distribution' || algorithm === 'balancing_info') {
                return;
            }

            const formattedName = algorithm.split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');

            chartData.push({
                name: formattedName,
                silhouette_score: metrics.silhouette_score,
                inertia: metrics.inertia ? metrics.inertia / Math.max(...Object.values(algorithm_results).map(m => m.inertia || 0)) : null // Normalize inertia
            });
        });
        return chartData;
    };

    // Add these console logs
    console.log('Objective:', objective);
    console.log('Algorithm Results:', algorithm_results);
    console.log('Chart Data:', prepareChartData());

    return (
        <div className="results-container">
            {/* Download Section */}
            <div className="download-section">
                <div className="download-buttons">
                    <button 
                        className="download-btn pdf"
                        onClick={() => handleDownload('pdf')}
                        disabled={downloadState.loading}
                    >
                        {downloadState.loading ? (
                            <Loader className="spin" size={18} />
                        ) : (
                            <FileDown size={18} />
                        )}
                        Download PDF
                    </button>
                    <button 
                        className="download-btn html"
                        onClick={() => handleDownload('html')}
                        disabled={downloadState.loading}
                    >
                        {downloadState.loading ? (
                            <Loader className="spin" size={18} />
                        ) : (
                            <FileText size={18} />
                        )}
                        Download HTML
                    </button>
                </div>
                {downloadState.error && (
                    <div className="download-error">
                        <AlertTriangle size={16} />
                        {downloadState.error}
                    </div>
                )}
            </div>

            {/* Dataset Information */}
            <section className="dataset-info">
                <h3>Dataset Information</h3>
                <div className="info-grid">
                    <div className="info-box">
                        <h4>
                            <Rows className="info-icon" size={18} />
                            Rows:
                        </h4>
                        <span>{data_shape.rows}</span>
                    </div>
                    <div className="info-box">
                        <h4>
                            <Columns className="info-icon" size={18} />
                            Columns:
                        </h4>
                        <span>{data_shape.columns}</span>
                    </div>
                </div>
            </section>

            {/* Algorithm Performance */}
            <section className="algorithm-performance">
                <h3>Algorithm Performance</h3>
                <div className="algorithms-grid">
                    {Object.entries(algorithm_results).map(([algorithm, metrics]) => {
                        if (algorithm !== 'data_distribution' && algorithm !== 'balancing_info') {
                            const filteredMetrics = filterMetrics(metrics);
                            return (
                                <div key={algorithm} className="algorithm-card">
                                    <h4>{algorithm.split('_').map(word => 
                                        word.charAt(0).toUpperCase() + word.slice(1)
                                    ).join(' ')}</h4>
                                    {Object.entries(filteredMetrics).map(([metric, value]) => {
                                        return (
                                            <div key={metric} className="metric-row">
                                                <span className="metric-name">
                                                    {getMetricIcon(metric)}
                                                    {metric.split('_').map(word => 
                                                        word.charAt(0).toUpperCase() + word.slice(1)
                                                    ).join(' ')}:
                                                </span>
                                                <span 
                                                    className="metric-value"
                                                    style={{ color: getMetricColor(value) }}
                                                >
                                                    {formatMetric(value)}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            );
                        }
                        return null;
                    })}
                </div>

                {/* Performance Chart */}
                <div className="chart-container">
                    {objective === 'clustering' ? (
                        <BarChart
                            width={800}
                            height={400}
                            data={prepareClusteringChartData()}
                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke={document.documentElement.getAttribute('data-theme') === 'dark' ? '#2d3748' : '#e0e0e0'} />
                            <XAxis dataKey="name" stroke={document.documentElement.getAttribute('data-theme') === 'dark' ? '#e0e0e0' : '#666'} />
                            <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                            <YAxis yAxisId="right" orientation="right" stroke="#4caf50" />
                            <Tooltip 
                                contentStyle={{
                                    backgroundColor: document.documentElement.getAttribute('data-theme') === 'dark' ? '#242c3d' : '#fff',
                                    border: `1px solid ${document.documentElement.getAttribute('data-theme') === 'dark' ? '#2d3748' : '#e0e0e0'}`,
                                    borderRadius: '4px'
                                }}
                                labelStyle={{
                                    color: document.documentElement.getAttribute('data-theme') === 'dark' ? '#e0e0e0' : '#666'
                                }}
                            />
                            <Legend wrapperStyle={{
                                color: document.documentElement.getAttribute('data-theme') === 'dark' ? '#e0e0e0' : '#666'
                            }} />
                            <Bar yAxisId="left" dataKey="silhouette_score" fill="#8884d8" name="Silhouette Score" />
                            <Bar yAxisId="right" dataKey="inertia" fill="#4caf50" name="Normalized Inertia" />
                        </BarChart>
                    ) : (
                        <BarChart
                            width={800}
                            height={400}
                            data={prepareChartData()}
                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis domain={[0, 'auto']} />
                            <Tooltip />
                            <Legend />
                            {objective === 'classification' ? (
                                <>
                                    <Bar dataKey="accuracy" fill="#4CAF50" name="Accuracy" />
                                    <Bar dataKey="precision" fill="#2196F3" name="Precision" />
                                    <Bar dataKey="recall" fill="#FFC107" name="Recall" />
                                    <Bar dataKey="f1_score" fill="#9C27B0" name="F1 Score" />
                                </>
                            ) : (
                                <>
                                    <Bar dataKey="r2_score" fill="#4CAF50" name="R² Score" />
                                    <Bar dataKey="mse" fill="#2196F3" name="MSE" />
                                    <Bar dataKey="rmse" fill="#FFC107" name="RMSE" />
                                </>
                            )}
                        </BarChart>
                    )}
                </div>

                {/* Add Clustering Explanation if needed */}
                {objective === 'clustering' && (
                    <>
                        <button 
                            className={`learn-more-button ${showClusteringMetrics ? 'expanded' : ''}`}
                            onClick={() => setShowClusteringMetrics(!showClusteringMetrics)}
                        >
                            <ChevronDown size={16} />
                            Want to learn more about these metrics?
                        </button>
                        
                        {showClusteringMetrics && (
                            <div className="clustering-explanation">
                                <h4>Understanding Clustering Metrics</h4>
                                <div className="metric-explanation">
                                    <div className="metric-explanation-item">
                                        <h5><Activity size={16} /> Silhouette Score</h5>
                                        <p>Measures how similar an object is to its own cluster compared to other clusters. 
                                           Ranges from -1 to 1, where higher values indicate better-defined clusters.</p>
                                    </div>
                                    <div className="metric-explanation-item">
                                        <h5><Layers size={16} /> Number of Clusters</h5>
                                        <p>The number of distinct groups the data has been divided into.</p>
                                    </div>
                                    <div className="metric-explanation-item">
                                        <h5><Target size={16} /> Inertia</h5>
                                        <p>Measures how compact the clusters are. Lower values indicate more compact clusters, 
                                           but this value depends on the dataset size and features.</p>
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                )}
            </section>

            {/* Preprocessing Details */}
            <section className="preprocessing-details">
                <h3>Preprocessing Details</h3>
                <div className="preprocessing-grid">
                    {/* Missing Values Handled */}
                    <div className="preprocessing-item">
                        <span className="preprocessing-label">
                            {getPreprocessingIcon('missing_values_handled')}
                            Missing Values Handled:
                        </span>
                        {renderValueWithIcon(preprocessing_info.missing_values_handled, preprocessing_info.missing_values_handled ? 'yes' : 'no')}
                    </div>

                    {/* Categorical Columns */}
                    <div className="preprocessing-item">
                        <span className="preprocessing-label">
                            {getPreprocessingIcon('categorical_columns')}
                            Categorical Columns:
                        </span>
                        <span className={`preprocessing-value ${preprocessing_info.categorical_columns.length === 0 ? 'none' : ''}`}>
                            {preprocessing_info.categorical_columns.length === 0 ? 'None' : preprocessing_info.categorical_columns.join(', ')}
                        </span>
                    </div>

                    {/* Scaled Features */}
                    <div className="preprocessing-item">
                        <span className="preprocessing-label">
                            {getPreprocessingIcon('scaled_features')}
                            Scaled Features:
                        </span>
                        {renderValueWithIcon(preprocessing_info.scaled_features, preprocessing_info.scaled_features ? 'yes' : 'no')}
                    </div>

                    {/* Target Type */}
                    <div className="preprocessing-item">
                        <span className="preprocessing-label">
                            {getPreprocessingIcon('target_type')}
                            Target Type:
                        </span>
                        <span className="preprocessing-value numeric">
                            {preprocessing_info.target_type}
                        </span>
                    </div>

                    {/* Small Dataset Handling */}
                    <div className="preprocessing-item">
                        <span className="preprocessing-label">
                            {getPreprocessingIcon('small_dataset_handling')}
                            Small Dataset Handling:
                        </span>
                        {renderValueWithIcon(preprocessing_info.small_dataset_handling, preprocessing_info.small_dataset_handling ? 'yes' : 'no')}
                    </div>
                </div>
            </section>

            {/* Class Distribution Section */}
            {results.target_classes && (
                <section className="target-classes">
                    <h3>
                        <Table className="section-icon" size={20} />
                        Class Distribution Analysis
                    </h3>
                    <div className="class-distribution-container">
                        {Object.entries(results.class_distribution || {}).map(([className, count]) => {
                            const total = Object.values(results.class_distribution).reduce((a, b) => a + b, 0);
                            const percentage = ((count / total) * 100).toFixed(1);
                            return (
                                <div key={className} className="class-stat-item">
                                    <div className="class-header">
                                        <span className="class-label">Class {className}</span>
                                        <span className="percentage">{percentage}%</span>
                                    </div>
                                    <div className="class-details">
                                        <div className="sample-count">{count}</div>
                                        <div className="sample-label">samples</div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </section>
            )}
        </div>
    );
}

export default ResultsDisplay;

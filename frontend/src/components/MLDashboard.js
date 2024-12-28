// src/components/MLDashboard.js
import React, { useState } from 'react';
import './MLDashboard.css';

function MLDashboard() {
  const [mlObjective, setMlObjective] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    // Send data to backend, or simulate response
    setResults({
      Accuracy: 0.85,
      Precision: 0.82,
      Recall: 0.88,
      'F1 Score': 0.85,
    });
  };

  const handleDownloadPDF = () => {
    console.log('Downloading PDF...');
  };

  return (
    <div className="ml-dashboard">
      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Upload CSV File</label>
            <input type="file" accept=".csv" onChange={handleFileChange} />
          </div>

          <div className="form-group">
            <label>ML Objective</label>
            <select value={mlObjective} onChange={(e) => setMlObjective(e.target.value)}>
              <option value="">Select ML Objective</option>
              <option value="regression">Regression</option>
              <option value="classification">Classification</option>
              <option value="clustering">Clustering</option>
            </select>
          </div>

          {(mlObjective === 'regression' || mlObjective === 'classification') && (
            <div className="form-group">
              <label>Target Column</label>
              <input
                type="text"
                placeholder="Enter target column name"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
              />
            </div>
          )}

          <button type="submit" className="btn btn-primary">
            Run Analysis
          </button>
        </form>

        {results && (
          <div className="results-section">
            <h2>Analysis Results</h2>
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results).map(([metric, value]) => (
                  <tr key={metric}>
                    <td>{metric}</td>
                    <td>{value.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button onClick={handleDownloadPDF} className="btn btn-download">
              Download PDF Report
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default MLDashboard;

// src/components/UploadForm.js
import React, { useState } from 'react';
import './UploadForm.css';  // <-- your custom CSS for styling

function UploadForm() {
  const [file, setFile] = useState(null);
  const [objective, setObjective] = useState('regression');
  const [target, setTarget] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a CSV file.');
      return;
    }

    const formData = new FormData();
    formData.append('datafile', file);
    formData.append('objective', objective);
    formData.append('target', target);

    try {
      const response = await fetch('http://localhost:8000/ml/process/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }

      const data = await response.json();
      setResults(data.metrics);
      setError(null);
    } catch (err) {
      setError(err.message);
      setResults(null);
    }
  };

  return (
    <div className="upload-form-container">
      <h2 className="upload-form-title">Upload Dataset</h2>
      <form className="upload-form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="file" className="form-label">
            CSV File
          </label>
          <input
            id="file"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label htmlFor="objective" className="form-label">
            ML Objective
          </label>
          <select
            id="objective"
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            className="form-select"
          >
            <option value="regression">Regression</option>
            <option value="classification">Classification</option>
            <option value="clustering">Clustering</option>
          </select>
        </div>

        {(objective === 'regression' || objective === 'classification') && (
          <div className="form-group">
            <label htmlFor="target" className="form-label">
              Target Column
            </label>
            <input
              id="target"
              type="text"
              placeholder="Enter target column name"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="form-input"
            />
          </div>
        )}

        <button type="submit" className="btn btn-primary">
          Run Analysis
        </button>
      </form>

      {error && <p className="error-text">{error}</p>}
      {results && (
        <div className="results-container">
          <h3>Analysis Results</h3>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default UploadForm;

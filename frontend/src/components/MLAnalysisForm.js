// src/components/MLAnalysisForm.js
import React, { useState, useEffect } from 'react';
import './MLAnalysisForm.css';
import { Upload, FileType, Brain, Loader2, History, ChevronDown } from 'lucide-react';
import { Analytics, AutoGraph, Schema } from '@mui/icons-material';
import api from '../services/api';
import ResultsDisplay from './ResultsDisplay';

function MLAnalysisForm({ onAnalysisComplete }) {
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [mlObjective, setMlObjective] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [columns, setColumns] = useState([]);
  const [previousUploads, setPreviousUploads] = useState([]);
  const [selectedPreviousFile, setSelectedPreviousFile] = useState(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);

  useEffect(() => {
    fetchPreviousUploads();
  }, []);

  const fetchPreviousUploads = async () => {
    try {
      const response = await api.getPreviousUploads();
      setPreviousUploads(response);
    } catch (err) {
      console.error('Error fetching previous uploads:', err);
    }
  };

  const handlePreviousFileSelect = async (dataset) => {
    setSelectedPreviousFile(dataset);
    setFileId(dataset.id);
    setColumns(dataset.columns || []);
    setFile(null);
  };

  const handleFileChange = async (event) => {
    if (event.target.files && event.target.files[0]) {
      const selectedFile = event.target.files[0];
      setFile(selectedFile);
      setError(null);
      setIsLoading(true);

      try {
        const response = await api.uploadFile(selectedFile);
        setFileId(response.file_id);
        setColumns(response.columns || []);
        setError(null);
      } catch (err) {
        setError('Error reading file columns: ' + err.message);
        console.error('File upload error:', err);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      console.log('Submitting analysis request...');
      const response = await api.analyzeData(fileId, mlObjective, targetColumn);
      console.log('Full API Response:', response);
      
      if (!response.id) {
        console.error('No analysis ID in response');
        throw new Error('Analysis ID is missing from the response');
      }

      const resultsWithId = {
        ...response.results,
        analysisId: response.id
      };
      console.log('Setting analysis results with ID:', resultsWithId);
      
      setAnalysisResults(resultsWithId);
      setShowResults(true);
    } catch (err) {
      const errorMessage = err.message || 'An error occurred during analysis';
      console.error('Analysis error:', errorMessage);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/analysis/upload_file/', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      // Handle success
    } catch (error) {
      console.error('Upload error:', error);
      // Handle error
    }
  };

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest('.previous-uploads')) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, []);

  return (
    <div className="analysis-form-container">
      <form onSubmit={handleSubmit} className="analysis-form">
        {error && <div className="error-message">{error}</div>}
        
        <div className="form-group">
          <label>
            <Upload className="input-icon" size={20} />
            Upload CSV File
          </label>
          <div className="file-input-wrapper">
            <FileType className="file-icon" size={20} />
            <input 
              type="file" 
              accept=".csv" 
              onChange={handleFileChange}
              required={!selectedPreviousFile}
            />
          </div>
          {file && <div className="file-name">{file.name}</div>}
        </div>

        {previousUploads.length > 0 && (
          <div className={`form-group previous-uploads ${isDropdownOpen ? 'open' : ''}`}>
            <label>
              <History className="input-icon" size={20} />
              Or Select Previous Upload
            </label>
            <div 
              className="previous-uploads-header"
              onClick={toggleDropdown}
            >
              <div className="header-content">
                <FileType size={16} />
                <span>
                  {selectedPreviousFile 
                    ? selectedPreviousFile.filename 
                    : 'Select a previous file'}
                </span>
              </div>
              <ChevronDown 
                size={16} 
                className="chevron-icon"
              />
            </div>
            <div className="previous-files-list">
              {previousUploads.map((dataset) => (
                <div
                  key={dataset.id}
                  className={`previous-file ${selectedPreviousFile?.id === dataset.id ? 'selected' : ''}`}
                  onClick={() => {
                    handlePreviousFileSelect(dataset);
                    setIsDropdownOpen(false);
                  }}
                >
                  <FileType size={16} />
                  <div className="file-info">
                    <span className="filename">{dataset.filename}</span>
                    <span className="upload-date">{dataset.uploaded_at}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="form-group">
          <label>
            <Brain className="input-icon" size={20} />
            ML Objective
          </label>
          <div className="select-wrapper">
            <select 
              value={mlObjective} 
              onChange={(e) => {
                setMlObjective(e.target.value);
                setTargetColumn(''); // Reset target column when objective changes
              }}
              required
            >
              <option value="">Select ML Objective</option>
              <option value="classification">
                <span className="option-content">
                  <Schema fontSize="small" /> Classification
                </span>
              </option>
              <option value="regression">
                <span className="option-content">
                  <AutoGraph fontSize="small" /> Regression
                </span>
              </option>
              <option value="clustering">
                <span className="option-content">
                  <Analytics fontSize="small" /> Clustering
                </span>
              </option>
            </select>
          </div>
        </div>

        {(mlObjective === 'regression' || mlObjective === 'classification') && columns.length > 0 && (
          <div className="form-group">
            <label>
              <AutoGraph className="input-icon" size={20} />
              Target Column
            </label>
            <div className="select-wrapper">
              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                required
                className="column-select"
              >
                <option value="">Select Target Column</option>
                {columns.map((column) => (
                  <option key={column} value={column} className="column-option">
                    {column}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}

        <button 
          type="submit" 
          disabled={isLoading || !fileId || !mlObjective || 
                   ((mlObjective === 'regression' || mlObjective === 'classification') && !targetColumn)} 
          className="btn btn-primary"
        >
          {isLoading ? (
            <>
              <Loader2 className="spin-icon" size={20} />
              {file && !fileId ? 'Uploading...' : 'Analyzing...'}
            </>
          ) : (
            <>
              <Brain size={20} />
              Run Analysis
            </>
          )}
        </button>
      </form>

      {showResults && (
        <ResultsDisplay 
          results={analysisResults} 
          objective={mlObjective}
          analysisId={analysisResults.analysisId}
        />
      )}
    </div>
  );
}

export default MLAnalysisForm;

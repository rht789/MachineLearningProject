// src/App.js
import React, { useState } from 'react';
import './App.css';
import { ThemeProvider } from './context/ThemeContext';

// Import your components
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import MLAnalysisForm from './components/MLAnalysisForm';
import ResultsDisplay from './components/ResultsDisplay';
// or MLDashboard, UploadForm, etc. — whichever ones you want to show

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [showResults, setShowResults] = useState(false);

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setShowResults(true);
  };

  return (
    <ThemeProvider>
      <div className="app-container">
        <Navbar />
        <main className="main-section">
          <h1 className="main-title">Machine Learning Analysis Dashboard</h1>
          <p className="main-subtitle">
            Upload your data, choose your objective, and get instant ML insights.
          </p>

          <MLAnalysisForm onAnalysisComplete={handleAnalysisComplete} />
          {showResults && analysisResults && <ResultsDisplay results={analysisResults} />}
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}

export default App;

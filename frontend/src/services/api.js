import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

const api = {
    uploadFile: async (file) => {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_URL}/analysis/upload_file/`, {
                method: 'POST',
                body: formData,
                credentials: 'include',
                headers: {
                    'Accept': 'application/json',
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to upload file');
            }

            const data = await response.json();
            if (!data.columns || data.columns.length === 0) {
                throw new Error('No columns found in the CSV file');
            }

            return {
                file_id: data.file_id,
                columns: data.columns,
                filename: data.filename,
                rowCount: data.row_count
            };
        } catch (error) {
            console.error('Upload error:', error);
            throw new Error(error.message || 'Failed to upload file');
        }
    },

    analyzeData: async (fileId, objective, targetColumn, algorithm = 'auto') => {
        try {
            console.log('Making analysis request with:', { fileId, objective, targetColumn, algorithm });
            
            const response = await axios.post(`${API_URL}/analysis/`, {
                file_id: fileId,
                objective,
                target_column: targetColumn,
                algorithm
            }, {
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                withCredentials: true
            });

            console.log('Raw API Response:', response.data);
            
            if (!response.data.id) {
                console.error('API Response missing ID:', response.data);
                throw new Error('Analysis ID missing from server response');
            }

            return {
                id: response.data.id,
                results: response.data.results
            };
        } catch (error) {
            console.error('Analysis API Error:', error.response?.data || error);
            throw new Error(error.response?.data?.error || 'Analysis failed');
        }
    },

    getPreviousAnalyses: async () => {
        try {
            const response = await axios.get(`${API_URL}/analysis/previous_analyses/`, {
                withCredentials: true
            });
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    },

    getPreviousUploads: async () => {
        try {
            const response = await fetch(`${API_URL}/analysis/previous_uploads/`, {
                credentials: 'include',
                headers: {
                    'Accept': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch previous uploads');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching previous uploads:', error);
            throw error;
        }
    },

    getAvailableAlgorithms: async () => {
        try {
            const response = await axios.get(`${API_URL}/analysis/available_algorithms/`);
            return response.data;
        } catch (error) {
            throw new Error('Failed to fetch available algorithms');
        }
    }
};

export default api; 
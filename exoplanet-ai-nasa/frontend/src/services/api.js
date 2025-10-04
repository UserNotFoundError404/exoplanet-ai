import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API_BASE = `${BACKEND_URL}/api`;

// Create axios instance with base configuration
const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 minutes for ML operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// System API
export const systemAPI = {
  getStatus: () => api.get('/'),
  
  // Get system health and statistics
  getSystemInfo: async () => {
    try {
      const response = await api.get('/');
      return response.data;
    } catch (error) {
      throw new Error('Failed to fetch system status');
    }
  }
};

// Data Loading API
export const dataAPI = {
  // Load NASA data from specified sources
  loadData: async (request) => {
    try {
      const response = await api.post('/data/load', request);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to load data');
    }
  },

  // Get data overview and statistics
  getOverview: async () => {
    try {
      const response = await api.get('/data/overview');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to get data overview');
    }
  }
};

// Machine Learning API
export const mlAPI = {
  // Train ML models
  trainModels: async (request) => {
    try {
      const response = await api.post('/ml/train', request);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to start model training');
    }
  },

  // Get training status
  getTrainingStatus: async () => {
    try {
      const response = await api.get('/ml/training-status');
      return response.data;
    } catch (error) {
      throw new Error('Failed to get training status');
    }
  },

  // Get available models
  getModels: async () => {
    try {
      const response = await api.get('/ml/models');
      return response.data;
    } catch (error) {
      throw new Error('Failed to get available models');
    }
  },

  // Make single prediction
  predict: async (request) => {
    try {
      const response = await api.post('/ml/predict', request);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to make prediction');
    }
  },

  // Make batch predictions
  predictBatch: async (request) => {
    try {
      const response = await api.post('/ml/predict-batch', request);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to make batch predictions');
    }
  },

  // Get model performance metrics
  getPerformance: async () => {
    try {
      const response = await api.get('/ml/performance');
      return response.data;
    } catch (error) {
      throw new Error('Failed to get model performance');
    }
  }
};

// Visualization API
export const vizAPI = {
  // Create custom visualization
  createVisualization: async (request) => {
    try {
      const response = await api.post('/visualizations/create', request);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to create visualization');
    }
  },

  // Get light curve for target
  getLightCurve: async (targetName, mission = 'TESS') => {
    try {
      const response = await api.get(`/visualizations/light-curve/${encodeURIComponent(targetName)}`, {
        params: { mission }
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to get light curve');
    }
  },

  // Get transit analysis for target
  getTransitAnalysis: async (targetName) => {
    try {
      const response = await api.get(`/visualizations/transit/${encodeURIComponent(targetName)}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to get transit analysis');
    }
  }
};

// Utility functions
export const utils = {
  // Format numbers for display
  formatNumber: (num, decimals = 2) => {
    if (num === null || num === undefined) return 'N/A';
    if (typeof num !== 'number') return num;
    
    if (Math.abs(num) >= 1e6) {
      return `${(num / 1e6).toFixed(decimals)}M`;
    } else if (Math.abs(num) >= 1e3) {
      return `${(num / 1e3).toFixed(decimals)}K`;
    }
    return num.toFixed(decimals);
  },

  // Format percentage
  formatPercentage: (num, decimals = 1) => {
    if (num === null || num === undefined) return 'N/A';
    return `${(num * 100).toFixed(decimals)}%`;
  },

  // Format scientific notation
  formatScientific: (num, decimals = 2) => {
    if (num === null || num === undefined) return 'N/A';
    if (typeof num !== 'number') return num;
    return num.toExponential(decimals);
  },

  // Get status color
  getStatusColor: (status) => {
    switch (status) {
      case 'operational':
      case 'completed':
      case 'success':
        return 'text-green-600';
      case 'in_progress':
      case 'training':
        return 'text-yellow-600';
      case 'error':
      case 'failed':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  }
};

// Export constants
export const CONSTANTS = {
  DATA_SOURCES: [
    'Kepler Confirmed Planets',
    'Kepler KOI Cumulative',
    'TESS Objects of Interest',
    'Planetary Systems Composite'
  ],
  
  ML_ALGORITHMS: [
    'Random Forest',
    'XGBoost',
    'SVM',
    'Logistic Regression',
    'Neural Network',
    'Extra Trees',
    'Gradient Boosting'
  ],

  EXOPLANET_TYPES: [
    'Hot Jupiter',
    'Warm Jupiter',
    'Cold Jupiter',
    'Super Earth',
    'Sub Neptune',
    'Neptune-like',
    'Terrestrial',
    'Mini Neptune',
    'Gas Giant',
    'Rocky Planet'
  ],

  FEATURE_COLUMNS: [
    'pl_rade',    // Planet radius (Earth radii)
    'pl_masse',   // Planet mass (Earth masses)
    'pl_orbper',  // Orbital period (days)
    'pl_orbsmax', // Semi-major axis (AU)
    'pl_orbeccen',// Eccentricity
    'pl_eqt',     // Equilibrium temperature (K)
    'st_rad',     // Stellar radius (Solar radii)
    'st_mass',    // Stellar mass (Solar masses)
    'st_teff',    // Stellar effective temperature (K)
    'st_met',     // Stellar metallicity
    'st_logg',    // Stellar surface gravity
    'sy_dist'     // System distance (parsecs)
  ],

  FEATURE_LABELS: {
    'pl_rade': 'Planet Radius (R⊕)',
    'pl_masse': 'Planet Mass (M⊕)',
    'pl_orbper': 'Orbital Period (days)',
    'pl_orbsmax': 'Semi-major Axis (AU)',
    'pl_orbeccen': 'Eccentricity',
    'pl_eqt': 'Equilibrium Temp. (K)',
    'st_rad': 'Stellar Radius (R☉)',
    'st_mass': 'Stellar Mass (M☉)',
    'st_teff': 'Stellar Temp. (K)',
    'st_met': 'Stellar Metallicity [Fe/H]',
    'st_logg': 'Stellar Surface Gravity',
    'sy_dist': 'Distance (parsecs)'
  }
};

export default api;
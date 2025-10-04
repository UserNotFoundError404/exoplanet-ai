import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import DataLoading from './components/DataLoading';
import './App.css';

// Placeholder components for other pages
const DataAnalysis = () => (
  <div className="bg-white rounded-lg border border-gray-200 p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">Data Analysis</h1>
    <p className="text-gray-600">Data analysis and exploration tools coming soon...</p>
  </div>
);

const MLTraining = () => (
  <div className="bg-white rounded-lg border border-gray-200 p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">ML Training</h1>
    <p className="text-gray-600">Machine learning model training interface coming soon...</p>
  </div>
);

const Predictions = () => (
  <div className="bg-white rounded-lg border border-gray-200 p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">Predictions</h1>
    <p className="text-gray-600">Exoplanet classification predictions coming soon...</p>
  </div>
);

const Visualizations = () => (
  <div className="bg-white rounded-lg border border-gray-200 p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">Visualizations</h1>
    <p className="text-gray-600">Interactive charts and visualizations coming soon...</p>
  </div>
);

const LightCurves = () => (
  <div className="bg-white rounded-lg border border-gray-200 p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">Light Curves</h1>
    <p className="text-gray-600">Light curve analysis and transit detection coming soon...</p>
  </div>
);

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="data" element={<DataLoading />} />
          <Route path="analysis" element={<DataAnalysis />} />
          <Route path="ml-training" element={<MLTraining />} />
          <Route path="predictions" element={<Predictions />} />
          <Route path="visualizations" element={<Visualizations />} />
          <Route path="light-curves" element={<LightCurves />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

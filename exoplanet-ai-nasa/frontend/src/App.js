import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import DataLoading from './components/DataLoading';
import DataAnalysis from './components/DataAnalysis';
import MLTraining from './components/MLTraining';
import Predictions from './components/Predictions';
import Visualizations from './components/Visualizations';
import LightCurves from './components/LightCurves';
// (optional if you added TransitAnalysis.jsx)
// import TransitAnalysis from './components/TransitAnalysis';

import './App.css';

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
          {/* Uncomment if you made TransitAnalysis */}
          {/* <Route path="transit" element={<TransitAnalysis />} /> */}
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

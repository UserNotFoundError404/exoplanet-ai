import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Download, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  Satellite,
  Globe,
  Activity,
  Clock
} from 'lucide-react';
import { dataAPI, CONSTANTS } from '../services/api';

const DataLoading = () => {
  const [selectedSources, setSelectedSources] = useState(['Kepler Confirmed Planets', 'TESS Objects of Interest']);
  const [limitPerSource, setLimitPerSource] = useState(2000);
  const [includeLightCurves, setIncludeLightCurves] = useState(false);
  const [includeTransitData, setIncludeTransitData] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadResult, setLoadResult] = useState(null);
  const [error, setError] = useState(null);
  const [loadProgress, setLoadProgress] = useState(0);

  const handleSourceToggle = (source) => {
    if (selectedSources.includes(source)) {
      setSelectedSources(selectedSources.filter(s => s !== source));
    } else {
      setSelectedSources([...selectedSources, source]);
    }
  };

  const handleLoadData = async () => {
    if (selectedSources.length === 0) {
      setError('Please select at least one data source');
      return;
    }

    setLoading(true);
    setError(null);
    setLoadResult(null);
    setLoadProgress(0);

    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setLoadProgress(prev => Math.min(prev + 10, 90));
    }, 500);

    try {
      const request = {
        data_sources: selectedSources,
        limit_per_source: limitPerSource,
        include_light_curves: includeLightCurves,
        include_transit_data: includeTransitData
      };

      const result = await dataAPI.loadData(request);
      setLoadResult(result);
      setLoadProgress(100);
      
      // Update sidebar counts if elements exist
      const dataCountEl = document.getElementById('data-count');
      if (dataCountEl) dataCountEl.textContent = result.total_records.toLocaleString();
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      clearInterval(progressInterval);
    }
  };

  const DataSourceCard = ({ source, selected, onToggle, description, icon: Icon }) => (
    <div 
      className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
        selected 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-200 hover:border-gray-300'
      }`}
      onClick={() => onToggle(source)}
    >
      <div className="flex items-start space-x-3">
        <div className={`p-2 rounded-lg ${selected ? 'bg-blue-100' : 'bg-gray-100'}`}>
          <Icon className={`h-5 w-5 ${selected ? 'text-blue-600' : 'text-gray-600'}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <h4 className={`font-medium ${selected ? 'text-blue-900' : 'text-gray-900'}`}>
              {source}
            </h4>
            {selected && <CheckCircle className="h-5 w-5 text-blue-600" />}
          </div>
          <p className="text-sm text-gray-600 mt-1">{description}</p>
        </div>
      </div>
    </div>
  );

  const sourceDescriptions = {
    'Kepler Confirmed Planets': 'Confirmed exoplanets discovered by the Kepler Space Telescope',
    'Kepler KOI Cumulative': 'Kepler Objects of Interest from the cumulative catalog',
    'TESS Objects of Interest': 'Candidate planets from the Transiting Exoplanet Survey Satellite',
    'Planetary Systems Composite': 'Comprehensive planetary system parameters'
  };

  const sourceIcons = {
    'Kepler Confirmed Planets': Satellite,
    'Kepler KOI Cumulative': Satellite,
    'TESS Objects of Interest': Globe,
    'Planetary Systems Composite': Activity
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Data Loading</h1>
        <p className="text-gray-600 mt-1">Load exoplanet data from NASA archives and databases</p>
      </div>

      {/* Progress Bar (when loading) */}
      {loading && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Loading data...</span>
            <span className="text-sm text-gray-500">{loadProgress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${loadProgress}%` }}
            ></div>
          </div>
          <p className="text-sm text-gray-500 mt-2">
            Fetching data from {selectedSources.length} source{selectedSources.length > 1 ? 's' : ''}...
          </p>
        </div>
      )}

      {/* Load Result */}
      {loadResult && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
            <div>
              <p className="font-medium text-green-800">Data loaded successfully!</p>
              <div className="mt-2 text-sm text-green-700">
                <p>• {loadResult.total_records.toLocaleString()} total records loaded</p>
                <p>• {loadResult.features_count} feature columns</p>
                <p>• {loadResult.sources_loaded.length} sources processed</p>
                <p>• Completed in {loadResult.load_time.toFixed(1)} seconds</p>
              </div>
            </div>
          </div>
          
          {/* Detailed breakdown */}
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(loadResult.records_per_source).map(([source, count]) => (
              <div key={source} className="flex justify-between items-center py-2 px-3 bg-white rounded border border-green-200">
                <span className="text-sm font-medium text-gray-700">{source}</span>
                <span className="text-sm text-green-600">{count.toLocaleString()} records</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
            <div>
              <p className="font-medium text-red-800">Error loading data</p>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Data Sources Selection */}
        <div className="lg:col-span-2">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Select Data Sources
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {CONSTANTS.DATA_SOURCES.map((source) => (
                <DataSourceCard
                  key={source}
                  source={source}
                  selected={selectedSources.includes(source)}
                  onToggle={handleSourceToggle}
                  description={sourceDescriptions[source]}
                  icon={sourceIcons[source]}
                />
              ))}
            </div>

            {/* Additional Options */}
            <div className="border-t border-gray-200 pt-6">
              <h4 className="text-md font-medium text-gray-900 mb-4">Additional Data Options</h4>
              
              <div className="space-y-3">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={includeLightCurves}
                    onChange={(e) => setIncludeLightCurves(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-3 text-sm text-gray-700">
                    Include light curve data (for first 10 targets)
                  </span>
                </label>
                
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={includeTransitData}
                    onChange={(e) => setIncludeTransitData(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-3 text-sm text-gray-700">
                    Include transit parameters (for first 10 targets)
                  </span>
                </label>
              </div>

              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Records per source (max: 10,000)
                </label>
                <input
                  type="number"
                  value={limitPerSource}
                  onChange={(e) => setLimitPerSource(Math.max(100, Math.min(10000, parseInt(e.target.value) || 2000)))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="100"
                  max="10000"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Configuration Panel */}
        <div className="space-y-6">
          {/* Load Configuration */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Load Configuration</h3>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Selected Sources</span>
                  <span className="font-medium">{selectedSources.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Records per Source</span>
                  <span className="font-medium">{limitPerSource.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Estimated Total</span>
                  <span className="font-medium text-blue-600">
                    ~{(selectedSources.length * limitPerSource).toLocaleString()}
                  </span>
                </div>
              </div>

              <div className="pt-4 border-t border-gray-200">
                <button
                  onClick={handleLoadData}
                  disabled={loading || selectedSources.length === 0}
                  className={`w-full flex items-center justify-center px-4 py-3 rounded-md font-medium transition-colors ${
                    loading || selectedSources.length === 0
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {loading ? (
                    <Loader2 className="animate-spin h-4 w-4 mr-2" />
                  ) : (
                    <Download className="h-4 w-4 mr-2" />
                  )}
                  {loading ? 'Loading Data...' : 'Load NASA Data'}
                </button>
              </div>
            </div>
          </div>

          {/* Information Panel */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-3">Data Sources Info</h3>
            
            <div className="space-y-3 text-sm text-blue-800">
              <div className="flex items-center">
                <Satellite className="h-4 w-4 mr-2" />
                <span>Kepler: 2009-2017 mission data</span>
              </div>
              <div className="flex items-center">
                <Globe className="h-4 w-4 mr-2" />
                <span>TESS: 2018-ongoing observations</span>
              </div>
              <div className="flex items-center">
                <Database className="h-4 w-4 mr-2" />
                <span>NASA Exoplanet Archive</span>
              </div>
              <div className="flex items-center">
                <Clock className="h-4 w-4 mr-2" />
                <span>Updated regularly from NASA</span>
              </div>
            </div>

            <div className="mt-4 p-3 bg-blue-100 rounded border border-blue-300">
              <p className="text-xs text-blue-700">
                <strong>Note:</strong> Loading large datasets may take several minutes. 
                Light curves and transit data are loaded in the background for performance.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataLoading;
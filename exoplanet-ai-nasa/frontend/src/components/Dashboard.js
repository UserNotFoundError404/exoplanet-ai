import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Brain, 
  Activity, 
  TrendingUp, 
  Clock,
  CheckCircle,
  AlertCircle,
  Zap,
  Satellite
} from 'lucide-react';
import { systemAPI, dataAPI, mlAPI } from '../services/api';

const Dashboard = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [dataOverview, setDataOverview] = useState(null);
  const [modelStats, setModelStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Fetch system status
      const status = await systemAPI.getSystemInfo();
      setSystemStatus(status);

      // Try to fetch data overview (may fail if no data loaded)
      try {
        const overview = await dataAPI.getOverview();
        setDataOverview(overview);
      } catch (err) {
        setDataOverview(null);
      }

      // Try to fetch model statistics
      try {
        const models = await mlAPI.getModels();
        setModelStats(models);
      } catch (err) {
        setModelStats(null);
      }

      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !systemStatus) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Loading dashboard...</span>
      </div>
    );
  }

  if (error && !systemStatus) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <AlertCircle className="h-5 w-5 text-red-400" />
          <span className="ml-2 text-red-800">Failed to load dashboard: {error}</span>
        </div>
      </div>
    );
  }

  const StatCard = ({ title, value, icon: Icon, color, description, trend }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold ${color || 'text-gray-900'}`}>{value}</p>
          {description && (
            <p className="text-sm text-gray-500 mt-1">{description}</p>
          )}
          {trend && (
            <div className="flex items-center mt-2">
              <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
              <span className="text-sm text-green-600">{trend}</span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-full ${color?.replace('text', 'bg')?.replace('900', '100')}`}>
          <Icon className={`h-6 w-6 ${color || 'text-gray-600'}`} />
        </div>
      </div>
    </div>
  );

  const formatNumber = (num) => {
    if (num === null || num === undefined) return '0';
    return num.toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard Overview</h1>
        <p className="text-gray-600 mt-1">NASA Stellar Data Analysis Platform Status</p>
      </div>

      {/* System Status Banner */}
      <div className={`p-4 rounded-lg border ${
        systemStatus?.status === 'operational' 
          ? 'bg-green-50 border-green-200' 
          : 'bg-red-50 border-red-200'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            {systemStatus?.status === 'operational' ? (
              <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
            ) : (
              <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
            )}
            <div>
              <p className={`font-medium ${
                systemStatus?.status === 'operational' ? 'text-green-800' : 'text-red-800'
              }`}>
                System {systemStatus?.status || 'Unknown'}
              </p>
              <p className={`text-sm ${
                systemStatus?.status === 'operational' ? 'text-green-600' : 'text-red-600'
              }`}>
                {systemStatus?.database_connected 
                  ? 'Database connected • NASA API configured' 
                  : 'Database connection issues'}
              </p>
            </div>
          </div>
          <div className="text-sm text-gray-500">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Main Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Data Records"
          value={formatNumber(systemStatus?.data_records || 0)}
          icon={Database}
          color="text-blue-600"
          description="Exoplanet entries loaded"
        />
        
        <StatCard
          title="Trained Models"
          value={formatNumber(systemStatus?.models_available || 0)}
          icon={Brain}
          color="text-purple-600"
          description="ML models available"
        />
        
        <StatCard
          title="Data Sources"
          value={systemStatus?.system_info?.supported_data_sources?.length || 0}
          icon={Satellite}
          color="text-green-600"
          description="NASA archives connected"
        />
        
        <StatCard
          title="Planet Types"
          value={systemStatus?.system_info?.exoplanet_types?.length || 0}
          icon={Activity}
          color="text-orange-600"
          description="Classification categories"
        />
      </div>

      {/* Detailed Information Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Data Overview */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Data Overview</h3>
            <Database className="h-5 w-5 text-blue-600" />
          </div>
          
          {dataOverview ? (
            <div className="space-y-3">
              <div className="flex justify-between items-center py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Total Records</span>
                <span className="font-medium">{formatNumber(dataOverview.total_records)}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Features</span>
                <span className="font-medium">{dataOverview.features?.length || 0}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Data Sources</span>
                <span className="font-medium">{Object.keys(dataOverview.data_sources || {}).length}</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-sm text-gray-600">Planet Types</span>
                <span className="font-medium">{Object.keys(dataOverview.planet_types || {}).length}</span>
              </div>
              
              {/* Top planet types */}
              {dataOverview.planet_types && (
                <div className="mt-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Most Common Types</h4>
                  <div className="space-y-1">
                    {Object.entries(dataOverview.planet_types)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 3)
                      .map(([type, count]) => (
                        <div key={type} className="flex justify-between text-sm">
                          <span className="text-gray-600">{type}</span>
                          <span className="font-medium">{count}</span>
                        </div>
                      ))
                    }
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-gray-500">
              <Database className="h-8 w-8 mb-2 opacity-50" />
              <p className="text-sm">No data loaded yet</p>
              <p className="text-xs">Load NASA data to see overview</p>
            </div>
          )}
        </div>

        {/* Model Performance */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Model Performance</h3>
            <Brain className="h-5 w-5 text-purple-600" />
          </div>
          
          {modelStats?.models && modelStats.models.length > 0 ? (
            <div className="space-y-3">
              {modelStats.models.slice(0, 4).map((model) => (
                <div key={model.id} className="flex justify-between items-center py-2 border-b border-gray-100">
                  <div>
                    <span className="text-sm font-medium text-gray-900">{model.model_type}</span>
                    <p className="text-xs text-gray-500">
                      {formatNumber(model.training_data_count)} samples
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-medium text-green-600">
                      {(model.accuracy * 100).toFixed(1)}%
                    </span>
                    <p className="text-xs text-gray-500">accuracy</p>
                  </div>
                </div>
              ))}
              
              <div className="mt-4 p-3 bg-purple-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-purple-800">Total Models</span>
                  <span className="text-lg font-bold text-purple-600">{modelStats.total}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-gray-500">
              <Brain className="h-8 w-8 mb-2 opacity-50" />
              <p className="text-sm">No models trained yet</p>
              <p className="text-xs">Train ML models to see performance</p>
            </div>
          )}
        </div>
      </div>

      {/* System Configuration */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">System Configuration</h3>
          <Zap className="h-5 w-5 text-yellow-600" />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Available Algorithms */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Available Algorithms</h4>
            <div className="space-y-1">
              {systemStatus?.system_info?.available_algorithms?.map((algorithm) => (
                <div key={algorithm} className="flex items-center text-sm text-gray-600">
                  <CheckCircle className="h-3 w-3 text-green-500 mr-2" />
                  {algorithm}
                </div>
              ))}
            </div>
          </div>
          
          {/* Data Sources */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Data Sources</h4>
            <div className="space-y-1">
              {systemStatus?.system_info?.supported_data_sources?.map((source) => (
                <div key={source} className="flex items-center text-sm text-gray-600">
                  <CheckCircle className="h-3 w-3 text-green-500 mr-2" />
                  {source}
                </div>
              ))}
            </div>
          </div>
          
          {/* System Status */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">System Status</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">NASA API</span>
                <span className={`px-2 py-1 rounded-full text-xs ${
                  systemStatus?.system_info?.nasa_api_configured 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {systemStatus?.system_info?.nasa_api_configured ? 'Configured' : 'Not Configured'}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Database</span>
                <span className={`px-2 py-1 rounded-full text-xs ${
                  systemStatus?.database_connected 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {systemStatus?.database_connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Last Data Load</span>
                <span className="text-gray-500 text-xs">
                  {systemStatus?.last_data_load 
                    ? new Date(systemStatus.last_data_load).toLocaleDateString()
                    : 'Never'
                  }
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button 
            className="flex items-center justify-center p-4 bg-white rounded-lg border border-gray-200 hover:shadow-md transition-shadow"
            onClick={() => window.location.href = '/data'}
          >
            <Database className="h-5 w-5 text-blue-600 mr-2" />
            <span className="font-medium text-gray-900">Load NASA Data</span>
          </button>
          
          <button 
            className="flex items-center justify-center p-4 bg-white rounded-lg border border-gray-200 hover:shadow-md transition-shadow"
            onClick={() => window.location.href = '/ml-training'}
            disabled={!dataOverview}
          >
            <Brain className="h-5 w-5 text-purple-600 mr-2" />
            <span className="font-medium text-gray-900">Train Models</span>
          </button>
          
          <button 
            className="flex items-center justify-center p-4 bg-white rounded-lg border border-gray-200 hover:shadow-md transition-shadow"
            onClick={() => window.location.href = '/visualizations'}
            disabled={!dataOverview}
          >
            <Activity className="h-5 w-5 text-green-600 mr-2" />
            <span className="font-medium text-gray-900">View Analysis</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { vizAPI } from '../services/api';
import { Activity, Search, Download, AlertCircle, Telescope, Zap } from 'lucide-react';

const LightCurves = () => {
  const [targetName, setTargetName] = useState('');
  const [mission, setMission] = useState('TESS');
  const [lightCurveData, setLightCurveData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);

  // Sample targets for quick access
  const sampleTargets = [
    { name: 'TOI-715', description: 'Super-Earth in habitable zone' },
    { name: 'WASP-96', description: 'Hot Jupiter with atmospheric features' },
    { name: 'K2-18', description: 'Sub-Neptune with water vapor' },
    { name: 'TRAPPIST-1', description: 'Seven Earth-sized planets' },
    { name: 'Kepler-452b', description: 'Earth\'s cousin' },
    { name: 'HD 209458', description: 'First transiting exoplanet' }
  ];

  useEffect(() => {
    // Load search history from localStorage
    const history = localStorage.getItem('lightcurve_history');
    if (history) {
      setSearchHistory(JSON.parse(history));
    }
  }, []);

  const handleSearch = async () => {
    if (!targetName.trim()) {
      setError('Please enter a target name');
      return;
    }

    setLoading(true);
    setError(null);
    setLightCurveData(null);

    try {
      const data = await vizAPI.getLightCurve(targetName.trim(), mission);
      setLightCurveData(data);
      
      // Add to search history
      const newHistory = [
        { name: targetName.trim(), mission, timestamp: new Date().toISOString() },
        ...searchHistory.filter(item => 
          !(item.name === targetName.trim() && item.mission === mission)
        )
      ].slice(0, 10); // Keep only last 10 searches
      
      setSearchHistory(newHistory);
      localStorage.setItem('lightcurve_history', JSON.stringify(newHistory));
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleTarget = (sampleName) => {
    setTargetName(sampleName);
  };

  const handleHistoryItem = (historyItem) => {
    setTargetName(historyItem.name);
    setMission(historyItem.mission);
  };

  const downloadLightCurve = () => {
    if (!lightCurveData) return;
    
    const csvContent = [
      'Time,Flux,Flux_Error',
      ...lightCurveData.time.map((time, index) => 
        `${time},${lightCurveData.flux[index]},${lightCurveData.flux_err?.[index] || ''}`
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${targetName}_lightcurve.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const renderLightCurve = () => {
    if (!lightCurveData || lightCurveData.type === 'error') {
      return (
        <div className="text-center py-8 text-gray-500">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>{lightCurveData?.message || 'No light curve data available'}</p>
        </div>
      );
    }

    try {
      const data = typeof lightCurveData.data === 'string' 
        ? JSON.parse(lightCurveData.data) 
        : lightCurveData.data;

      const layout = {
        ...data.layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif', size: 12 },
        margin: { t: 50, r: 50, b: 80, l: 80 },
        xaxis: {
          ...data.layout?.xaxis,
          title: 'Time (BJD - 2457000)',
          gridcolor: '#e5e7eb'
        },
        yaxis: {
          ...data.layout?.yaxis,
          title: 'Relative Flux',
          gridcolor: '#e5e7eb'
        }
      };

      return (
        <Plot
          data={data.data}
          layout={layout}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
          }}
          className="w-full"
          style={{ height: '500px' }}
        />
      );
    } catch (err) {
      return (
        <div className="text-center py-8 text-red-500">
          <AlertCircle className="h-12 w-12 mx-auto mb-4" />
          <p>Error rendering light curve: {err.message}</p>
        </div>
      );
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <Activity className="h-8 w-8 text-green-600" />
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Light Curve Analysis</h1>
          <p className="text-gray-600">Analyze stellar brightness variations and transit signals</p>
        </div>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Search Panel */}
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Search className="h-5 w-5" />
                <span>Target Search</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="target-name">Target Name</Label>
                <Input
                  id="target-name"
                  placeholder="e.g., TOI-715, WASP-96b"
                  value={targetName}
                  onChange={(e) => setTargetName(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                />
              </div>

              <div>
                <Label htmlFor="mission">Mission</Label>
                <Select value={mission} onValueChange={setMission}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="TESS">TESS</SelectItem>
                    <SelectItem value="Kepler">Kepler</SelectItem>
                    <SelectItem value="K2">K2</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button
                onClick={handleSearch}
                disabled={loading}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Telescope className="mr-2 h-4 w-4 animate-pulse" />
                    Loading...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    Search
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Sample Targets */}
          <Card>
            <CardHeader>
              <CardTitle>Sample Targets</CardTitle>
              <CardDescription>
                Try these well-known exoplanets
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {sampleTargets.map((target) => (
                  <Button
                    key={target.name}
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start text-left h-auto p-2"
                    onClick={() => handleSampleTarget(target.name)}
                  >
                    <div>
                      <p className="font-medium text-sm">{target.name}</p>
                      <p className="text-xs text-gray-500">{target.description}</p>
                    </div>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Search History */}
          {searchHistory.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Recent Searches</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {searchHistory.map((item, index) => (
                    <Button
                      key={index}
                      variant="ghost"
                      size="sm"
                      className="w-full justify-start text-left"
                      onClick={() => handleHistoryItem(item)}
                    >
                      <div className="flex items-center justify-between w-full">
                        <span className="text-sm">{item.name}</span>
                        <Badge variant="outline" className="text-xs">
                          {item.mission}
                        </Badge>
                      </div>
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Light Curve Display */}
        <div className="lg:col-span-3">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Activity className="h-5 w-5" />
                  <span>Light Curve</span>
                  {targetName && (
                    <Badge variant="outline">{targetName} ({mission})</Badge>
                  )}
                </div>
                {lightCurveData && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadLightCurve}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download CSV
                  </Button>
                )}
              </CardTitle>
              <CardDescription>
                Stellar brightness measurements over time showing potential transit events
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <Telescope className="h-12 w-12 mx-auto mb-4 animate-pulse text-gray-400" />
                    <p className="text-gray-600">Loading light curve data...</p>
                    <p className="text-sm text-gray-500 mt-2">
                      Fetching data from {mission} archive
                    </p>
                  </div>
                </div>
              ) : lightCurveData ? (
                <div className="space-y-4">
                  {renderLightCurve()}
                  
                  {/* Light Curve Statistics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
                    <div className="text-center">
                      <p className="text-lg font-bold text-blue-600">
                        {lightCurveData.time?.length || 0}
                      </p>
                      <p className="text-sm text-gray-600">Data Points</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-bold text-green-600">
                        {lightCurveData.mission || mission}
                      </p>
                      <p className="text-sm text-gray-600">Mission</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-bold text-purple-600">
                        {lightCurveData.cadence || 'N/A'}
                      </p>
                      <p className="text-sm text-gray-600">Cadence</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-bold text-orange-600">
                        {lightCurveData.sector || 'N/A'}
                      </p>
                      <p className="text-sm text-gray-600">Sector/Quarter</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <Activity className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">No Light Curve Loaded</h3>
                  <p className="text-sm mb-4">
                    Enter a target name and click search to view light curve data
                  </p>
                  <div className="flex items-center justify-center space-x-4 text-xs">
                    <div className="flex items-center space-x-1">
                      <Zap className="h-3 w-3" />
                      <span>Transit Detection</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Activity className="h-3 w-3" />
                      <span>Variability Analysis</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Telescope className="h-3 w-3" />
                      <span>Multi-Mission Data</span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default LightCurves;

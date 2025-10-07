import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Separator } from './ui/separator';
import { vizAPI } from '../services/api';
import { Orbit, Search, Download, AlertCircle, Clock, Ruler } from 'lucide-react';

const TransitAnalysis = () => {
  const [targetName, setTargetName] = useState('');
  const [transitData, setTransitData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Sample transit targets
  const sampleTargets = [
    { name: 'HD 209458 b', description: 'First detected transiting exoplanet' },
    { name: 'WASP-12 b', description: 'Ultra-hot Jupiter' },
    { name: 'TRAPPIST-1 b', description: 'Rocky planet in multi-planet system' },
    { name: 'K2-18 b', description: 'Sub-Neptune with atmospheric water' },
    { name: 'TOI-715 b', description: 'Super-Earth in habitable zone' }
  ];

  const handleSearch = async () => {
    if (!targetName.trim()) {
      setError('Please enter a target name');
      return;
    }

    setLoading(true);
    setError(null);
    setTransitData(null);

    try {
      const data = await vizAPI.getTransitAnalysis(targetName.trim());
      setTransitData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleTarget = (sampleName) => {
    setTargetName(sampleName);
  };

  const downloadTransitData = () => {
    if (!transitData) return;
    
    const csvContent = [
      'Parameter,Value,Unit',
      `Period,${transitData.period || 'N/A'},days`,
      `Duration,${transitData.duration || 'N/A'},hours`,
      `Depth,${transitData.depth || 'N/A'},ppm`,
      `Epoch,${transitData.epoch || 'N/A'},BJD`,
      `Impact Parameter,${transitData.impact_parameter || 'N/A'},`,
      `Planet Radius,${transitData.planet_radius || 'N/A'},Earth radii`,
      `Semi-major Axis,${transitData.semi_major_axis || 'N/A'},AU`
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${targetName}_transit_analysis.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const renderTransitChart = () => {
    if (!transitData || transitData.type === 'error') {
      return (
        <div className="text-center py-8 text-gray-500">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>{transitData?.message || 'No transit data available'}</p>
        </div>
      );
    }

    try {
      const data = typeof transitData.data === 'string' 
        ? JSON.parse(transitData.data) 
        : transitData.data;

      const layout = {
        ...data.layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif', size: 12 },
        margin: { t: 50, r: 50, b: 80, l: 80 }
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
          style={{ height: '400px' }}
        />
      );
    } catch (err) {
      return (
        <div className="text-center py-8 text-red-500">
          <AlertCircle className="h-12 w-12 mx-auto mb-4" />
          <p>Error rendering transit chart: {err.message}</p>
        </div>
      );
    }
  };

  const formatValue = (value, unit = '') => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') {
      return `${value.toFixed(4)} ${unit}`.trim();
    }
    return `${value} ${unit}`.trim();
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <Orbit className="h-8 w-8 text-orange-600" />
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Transit Analysis</h1>
          <p className="text-gray-600">Analyze planetary transit events and orbital parameters</p>
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
        <div className="lg:col-span-1">
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
                  placeholder="e.g., HD 209458 b"
                  value={targetName}
                  onChange={(e) => setTargetName(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                />
              </div>

              <Button
                onClick={handleSearch}
                disabled={loading}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Orbit className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    Analyze Transit
                  </>
                )}
              </Button>

              <Separator />

              <div>
                <h3 className="font-medium mb-2">Sample Targets</h3>
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
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Transit Analysis Display */}
        <div className="lg:col-span-3">
          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="lightcurve">Light Curve</TabsTrigger>
              <TabsTrigger value="parameters">Parameters</TabsTrigger>
            </TabsList>

            <TabsContent value="overview">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Orbit className="h-5 w-5" />
                      <span>Transit Overview</span>
                      {targetName && <Badge variant="outline">{targetName}</Badge>}
                    </div>
                    {transitData && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={downloadTransitData}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download
                      </Button>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="flex items-center justify-center py-12">
                      <div className="text-center">
                        <Orbit className="h-12 w-12 mx-auto mb-4 animate-spin text-gray-400" />
                        <p className="text-gray-600">Analyzing transit data...</p>
                      </div>
                    </div>
                  ) : transitData ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <Card className="border-blue-200 bg-blue-50">
                        <CardContent className="p-4 text-center">
                          <Clock className="h-8 w-8 mx-auto mb-2 text-blue-600" />
                          <p className="text-2xl font-bold text-blue-600">
                            {formatValue(transitData.period, 'days')}
                          </p>
                          <p className="text-sm text-gray-600">Orbital Period</p>
                        </CardContent>
                      </Card>

                      <Card className="border-green-200 bg-green-50">
                        <CardContent className="p-4 text-center">
                          <Clock className="h-8 w-8 mx-auto mb-2 text-green-600" />
                          <p className="text-2xl font-bold text-green-600">
                            {formatValue(transitData.duration, 'hrs')}
                          </p>
                          <p className="text-sm text-gray-600">Transit Duration</p>
                        </CardContent>
                      </Card>

                      <Card className="border-purple-200 bg-purple-50">
                        <CardContent className="p-4 text-center">
                          <Ruler className="h-8 w-8 mx-auto mb-2 text-purple-600" />
                          <p className="text-2xl font-bold text-purple-600">
                            {formatValue(transitData.depth, 'ppm')}
                          </p>
                          <p className="text-sm text-gray-600">Transit Depth</p>
                        </CardContent>
                      </Card>

                      <Card className="border-orange-200 bg-orange-50">
                        <CardContent className="p-4 text-center">
                          <Orbit className="h-8 w-8 mx-auto mb-2 text-orange-600" />
                          <p className="text-2xl font-bold text-orange-600">
                            {formatValue(transitData.planet_radius, 'R⊕')}
                          </p>
                          <p className="text-sm text-gray-600">Planet Radius</p>
                        </CardContent>
                      </Card>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <Orbit className="h-16 w-16 mx-auto mb-4 opacity-50" />
                      <h3 className="text-lg font-medium mb-2">No Transit Analysis</h3>
                      <p className="text-sm">
                        Enter a target name to analyze transit parameters and orbital characteristics
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="lightcurve">
              <Card>
                <CardHeader>
                  <CardTitle>Phase-Folded Light Curve</CardTitle>
                  <CardDescription>
                    Transit signal folded on the orbital period
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="flex items-center justify-center py-12">
                      <Orbit className="h-8 w-8 animate-spin text-gray-400" />
                    </div>
                  ) : (
                    renderTransitChart()
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="parameters">
              <Card>
                <CardHeader>
                  <CardTitle>Transit Parameters</CardTitle>
                  <CardDescription>
                    Detailed orbital and physical parameters
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {transitData ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h3 className="font-medium mb-3">Orbital Parameters</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center py-2 border-b">
                            <span className="text-sm text-gray-600">Period</span>
                            <span className="font-medium">{formatValue(transitData.period, 'days')}</span>
                          </div>
                          <div className="flex justify-between items-center py-2 border-b">
                            <span className="text-sm text-gray-600">Epoch (T₀)</span>
                            <span className="font-medium">{formatValue(transitData.epoch, 'BJD')}</span>
                          </div>
                          <div className="flex justify-between items-center py-2 border-b">
                            <span className="text-sm text-gray-600">Semi-major Axis</span>
                            <span className="font-medium">{formatValue(transitData.semi_major_axis, 'AU')}</span>
                          </div>
                          <div className="flex justify-between items-center py-2 border-b">
                            <span className="text-sm text-gray-600">Eccentricity</span>
                            <span className="font-medium">{formatValue(transitData.eccentricity)}</span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h3 className="font-medium mb-3">Transit Properties</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center py-2 border-b">
                            <span className="text-sm text-gray-600">Duration</span>
                            <span className="font-medium">{formatValue(transitData.duration, 'hours')}</span>
                          </div>
                          <div className="flex justify-between items-center py-2 border-b">
                            <span className="text-sm text-gray-600">Depth</span>
                            <span className="font-medium">{formatValue(transitData.depth, 'ppm')}</span>
                          </div>
                          <div className="flex justify-between items-center py-2 border-b">
                            <span className="text-sm text-gray-600">Impact Parameter</span>
                            <span className="font-medium">{formatValue(transitData.impact_parameter)}</span>
                          </div>
                          <div className="flex justify-between items-center py-2 border-b">
                            <span className="text-sm text-gray-600">Planet Radius</span>
                            <span className="font-medium">{formatValue(transitData.planet_radius, 'R⊕')}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <p>No parameter data available</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default TransitAnalysis;

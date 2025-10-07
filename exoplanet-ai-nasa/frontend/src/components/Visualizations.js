import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';
import { Separator } from './ui/separator';
import { vizAPI, dataAPI, mlAPI, CONSTANTS } from '../services/api';
import { BarChart3, PieChart, TrendingUp, Scatter, AlertCircle, RefreshCw, Download } from 'lucide-react';

const Visualizations = () => {
  const [dataOverview, setDataOverview] = useState(null);
  const [modelPerformance, setModelPerformance] = useState(null);
  const [customChart, setCustomChart] = useState(null);
  const [chartConfig, setChartConfig] = useState({
    chart_type: 'correlation_heatmap',
    x_feature: 'pl_rade',
    y_feature: 'pl_masse'
  });
  const [loading, setLoading] = useState({
    overview: false,
    performance: false,
    custom: false
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDataOverview();
    loadModelPerformance();
  }, []);

  const loadDataOverview = async () => {
    setLoading(prev => ({ ...prev, overview: true }));
    try {
      const overview = await dataAPI.getOverview();
      setDataOverview(overview);
    } catch (err) {
      setError('Failed to load data overview');
    } finally {
      setLoading(prev => ({ ...prev, overview: false }));
    }
  };

  const loadModelPerformance = async () => {
    setLoading(prev => ({ ...prev, performance: true }));
    try {
      const performance = await mlAPI.getPerformance();
      setModelPerformance(performance);
    } catch (err) {
      console.error('Failed to load model performance:', err);
    } finally {
      setLoading(prev => ({ ...prev, performance: false }));
    }
  };

  const createCustomVisualization = async () => {
    setLoading(prev => ({ ...prev, custom: true }));
    setError(null);
    
    try {
      const request = {
        chart_type: chartConfig.chart_type,
        parameters: {
          x_feature: chartConfig.x_feature,
          y_feature: chartConfig.y_feature
        }
      };

      const result = await vizAPI.createVisualization(request);
      setCustomChart(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, custom: false }));
    }
  };

  const downloadChart = (plotData, filename) => {
    // Convert plot to image and download
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(JSON.stringify(plotData)));
    element.setAttribute('download', `${filename}.json`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const renderPlotlyChart = (chartData, title = '') => {
    if (!chartData || chartData.type === 'error') {
      return (
        <div className="text-center py-8 text-gray-500">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>{chartData?.message || 'No data available for visualization'}</p>
        </div>
      );
    }

    try {
      const data = typeof chartData.data === 'string' ? JSON.parse(chartData.data) : chartData.data;
      const layout = {
        ...data.layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif', size: 12 },
        margin: { t: 50, r: 50, b: 50, l: 50 }
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
          <p>Error rendering chart: {err.message}</p>
        </div>
      );
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <BarChart3 className="h-8 w-8 text-indigo-600" />
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Data Visualizations</h1>
          <p className="text-gray-600">Interactive charts and analysis visualizations</p>
        </div>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">{error}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Data Overview</TabsTrigger>
          <TabsTrigger value="performance">Model Performance</TabsTrigger>
          <TabsTrigger value="custom">Custom Charts</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">Dataset Overview</h2>
              <Button
                variant="outline"
                onClick={loadDataOverview}
                disabled={loading.overview}
              >
                {loading.overview ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
              </Button>
            </div>

            {dataOverview && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <Card>
                  <CardContent className="p-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-600">{dataOverview.total_records}</p>
                      <p className="text-sm text-gray-600">Total Records</p>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-600">{dataOverview.features?.length || 0}</p>
                      <p className="text-sm text-gray-600">Features</p>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-600">
                        {Object.keys(dataOverview.data_sources || {}).length}
                      </p>
                      <p className="text-sm text-gray-600">Data Sources</p>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-orange-600">
                        {Object.keys(dataOverview.planet_types || {}).length}
                      </p>
                      <p className="text-sm text-gray-600">Planet Types</p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Dataset Overview Chart</span>
                  {dataOverview?.visualization && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => downloadChart(dataOverview.visualization, 'overview')}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loading.overview ? (
                  <div className="flex items-center justify-center py-12">
                    <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
                  </div>
                ) : (
                  renderPlotlyChart(dataOverview?.visualization, 'Dataset Overview')
                )}
              </CardContent>
            </Card>

            {dataOverview && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Data Sources Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {Object.entries(dataOverview.data_sources || {}).map(([source, count]) => (
                        <div key={source} className="flex justify-between items-center">
                          <span className="text-sm">{source}</span>
                          <Badge variant="outline">{count}</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Planet Types Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {Object.entries(dataOverview.planet_types || {}).map(([type, count]) => (
                        <div key={type} className="flex justify-between items-center">
                          <span className="text-sm">{type}</span>
                          <Badge variant="outline">{count}</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent value="performance">
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">Model Performance</h2>
              <Button
                variant="outline"
                onClick={loadModelPerformance}
                disabled={loading.performance}
              >
                {loading.performance ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
              </Button>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Performance Comparison</span>
                  {modelPerformance?.visualization && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => downloadChart(modelPerformance.visualization, 'performance')}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loading.performance ? (
                  <div className="flex items-center justify-center py-12">
                    <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
                  </div>
                ) : (
                  renderPlotlyChart(modelPerformance?.visualization, 'Model Performance')
                )}
              </CardContent>
            </Card>

            {modelPerformance?.models && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {modelPerformance.models.map((model, index) => (
                  <Card key={index}>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg">{model.model_type}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Accuracy</span>
                          <Badge variant="outline">
                            {(model.performance_metrics.accuracy * 100).toFixed(1)}%
                          </Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Precision</span>
                          <Badge variant="outline">
                            {(model.performance_metrics.precision * 100).toFixed(1)}%
                          </Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Recall</span>
                          <Badge variant="outline">
                            {(model.performance_metrics.recall * 100).toFixed(1)}%
                          </Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">F1 Score</span>
                          <Badge variant="outline">
                            {(model.performance_metrics.f1_score * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent value="custom">
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Custom Visualizations</h2>

            <Card>
              <CardHeader>
                <CardTitle>Chart Configuration</CardTitle>
                <CardDescription>
                  Create custom visualizations from your data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label htmlFor="chart-type">Chart Type</Label>
                    <Select
                      value={chartConfig.chart_type}
                      onValueChange={(value) => 
                        setChartConfig(prev => ({ ...prev, chart_type: value }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="correlation_heatmap">Correlation Heatmap</SelectItem>
                        <SelectItem value="discovery_timeline">Discovery Timeline</SelectItem>
                        <SelectItem value="planet_classification">Planet Classification</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {chartConfig.chart_type === 'planet_classification' && (
                    <>
                      <div>
                        <Label htmlFor="x-feature">X-Axis Feature</Label>
                        <Select
                          value={chartConfig.x_feature}
                          onValueChange={(value) => 
                            setChartConfig(prev => ({ ...prev, x_feature: value }))
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {CONSTANTS.FEATURE_COLUMNS.map(feature => (
                              <SelectItem key={feature} value={feature}>
                                {CONSTANTS.FEATURE_LABELS[feature] || feature}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label htmlFor="y-feature">Y-Axis Feature</Label>
                        <Select
                          value={chartConfig.y_feature}
                          onValueChange={(value) => 
                            setChartConfig(prev => ({ ...prev, y_feature: value }))
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {CONSTANTS.FEATURE_COLUMNS.map(feature => (
                              <SelectItem key={feature} value={feature}>
                                {CONSTANTS.FEATURE_LABELS[feature] || feature}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </>
                  )}
                </div>

                <Separator />

                <Button
                  onClick={createCustomVisualization}
                  disabled={loading.custom}
                  className="w-full"
                >
                  {loading.custom ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Creating Visualization...
                    </>
                  ) : (
                    <>
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Create Visualization
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Custom Chart</span>
                  {customChart && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => downloadChart(customChart, 'custom-chart')}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loading.custom ? (
                  <div className="flex items-center justify-center py-12">
                    <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
                  </div>
                ) : customChart ? (
                  renderPlotlyChart(customChart, 'Custom Visualization')
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <Scatter className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Configure and create a custom visualization above</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Visualizations;

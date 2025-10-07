import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Separator } from './ui/separator';
import { dataAPI, vizAPI } from '../services/api';
import { BarChart3, RefreshCw, AlertCircle, TrendingUp, PieChart } from 'lucide-react';

function DataAnalysis() {
  const [dataOverview, setDataOverview] = useState(null);
  const [correlationChart, setCorrelationChart] = useState(null);
  const [timelineChart, setTimelineChart] = useState(null);
  const [loading, setLoading] = useState({
    overview: false,
    correlation: false,
    timeline: false
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDataOverview();
    loadCorrelationAnalysis();
    loadDiscoveryTimeline();
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

  const loadCorrelationAnalysis = async () => {
    setLoading(prev => ({ ...prev, correlation: true }));
    try {
      const chart = await vizAPI.createVisualization({
        chart_type: 'correlation_heatmap'
      });
      setCorrelationChart(chart);
    } catch (err) {
      console.error('Failed to load correlation analysis:', err);
    } finally {
      setLoading(prev => ({ ...prev, correlation: false }));
    }
  };

  const loadDiscoveryTimeline = async () => {
    setLoading(prev => ({ ...prev, timeline: true }));
    try {
      const chart = await vizAPI.createVisualization({
        chart_type: 'discovery_timeline'
      });
      setTimelineChart(chart);
    } catch (err) {
      console.error('Failed to load discovery timeline:', err);
    } finally {
      setLoading(prev => ({ ...prev, timeline: false }));
    }
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

  const refreshAll = () => {
    loadDataOverview();
    loadCorrelationAnalysis();
    loadDiscoveryTimeline();
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <TrendingUp className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Data Analysis</h1>
            <p className="text-gray-600">Statistical analysis and insights from exoplanet data</p>
          </div>
        </div>
        <Button variant="outline" onClick={refreshAll}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh All
        </Button>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">{error}</AlertDescription>
        </Alert>
      )}

      {/* Summary Cards */}
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

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Dataset Overview</TabsTrigger>
          <TabsTrigger value="correlation">Correlation Analysis</TabsTrigger>
          <TabsTrigger value="timeline">Discovery Timeline</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <PieChart className="h-5 w-5" />
                <span>Dataset Overview</span>
              </CardTitle>
              <CardDescription>
                Comprehensive overview of the loaded exoplanet dataset
              </CardDescription>
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
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Data Sources</CardTitle>
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
                  <CardTitle>Planet Types</CardTitle>
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
        </TabsContent>

        <TabsContent value="correlation">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <BarChart3 className="h-5 w-5" />
                <span>Feature Correlation Analysis</span>
              </CardTitle>
              <CardDescription>
                Correlation matrix showing relationships between exoplanet features
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading.correlation ? (
                <div className="flex items-center justify-center py-12">
                  <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
                </div>
              ) : (
                renderPlotlyChart(correlationChart, 'Correlation Heatmap')
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timeline">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="h-5 w-5" />
                <span>Discovery Timeline</span>
              </CardTitle>
              <CardDescription>
                Timeline of exoplanet discoveries over the years
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading.timeline ? (
                <div className="flex items-center justify-center py-12">
                  <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
                </div>
              ) : (
                renderPlotlyChart(timelineChart, 'Discovery Timeline')
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default DataAnalysis;

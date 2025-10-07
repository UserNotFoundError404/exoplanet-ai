import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { Textarea } from './ui/textarea';
import { mlAPI, CONSTANTS, utils } from '../services/api';
import { Target, Zap, Upload, Download, AlertCircle, CheckCircle, Brain } from 'lucide-react';

const Predictions = () => {
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [singleFeatures, setSingleFeatures] = useState({});
  const [batchData, setBatchData] = useState('');
  const [singlePrediction, setSinglePrediction] = useState(null);
  const [batchPredictions, setBatchPredictions] = useState(null);
  const [loading, setLoading] = useState({ single: false, batch: false });
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    fetchAvailableModels();
    initializeSingleFeatures();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const models = await mlAPI.getModels();
      setAvailableModels(models.models || []);
      if (models.models && models.models.length > 0) {
        setSelectedModel(models.models[0].id);
      }
    } catch (err) {
      setError('Failed to fetch available models');
    }
  };

  const initializeSingleFeatures = () => {
    const features = {};
    CONSTANTS.FEATURE_COLUMNS.forEach(feature => {
      features[feature] = '';
    });
    setSingleFeatures(features);
  };

  const handleSingleFeatureChange = (feature, value) => {
    setSingleFeatures(prev => ({
      ...prev,
      [feature]: value
    }));
  };

  const handleSinglePrediction = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    // Validate features
    const numericFeatures = {};
    let hasError = false;

    Object.entries(singleFeatures).forEach(([key, value]) => {
      if (value !== '') {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
          setError(`Invalid value for ${CONSTANTS.FEATURE_LABELS[key] || key}`);
          hasError = true;
          return;
        }
        numericFeatures[key] = numValue;
      }
    });

    if (hasError) return;
    if (Object.keys(numericFeatures).length === 0) {
      setError('Please provide at least one feature value');
      return;
    }

    setLoading(prev => ({ ...prev, single: true }));
    setError(null);
    setSuccess(null);

    try {
      const request = {
        model_id: selectedModel,
        features: numericFeatures,
        use_ensemble: false
      };

      const result = await mlAPI.predict(request);
      setSinglePrediction(result);
      setSuccess('Prediction completed successfully!');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, single: false }));
    }
  };

  const handleBatchPrediction = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    if (!batchData.trim()) {
      setError('Please provide batch data');
      return;
    }

    setLoading(prev => ({ ...prev, batch: true }));
    setError(null);
    setSuccess(null);

    try {
      // Parse CSV-like data
      const lines = batchData.trim().split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      const data = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
          const value = parseFloat(values[index]);
          if (!isNaN(value)) {
            row[header] = value;
          }
        });
        if (Object.keys(row).length > 0) {
          data.push(row);
        }
      }

      if (data.length === 0) {
        setError('No valid data rows found');
        return;
      }

      const request = {
        model_id: selectedModel,
        data: data,
        use_ensemble: true
      };

      const result = await mlAPI.predictBatch(request);
      setBatchPredictions(result);
      setSuccess(`Batch prediction completed for ${data.length} samples!`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, batch: false }));
    }
  };

  const generateSampleData = () => {
    const sampleData = [
      'pl_rade,pl_masse,pl_orbper,st_teff,st_mass',
      '1.2,1.5,365.25,5778,1.0',
      '11.2,317.8,11.86,5778,1.0',
      '0.8,0.6,87.97,5778,1.0'
    ].join('\n');
    setBatchData(sampleData);
  };

  const exportResults = () => {
    if (!batchPredictions) return;
    
    const csvContent = [
      'Index,Prediction,Confidence',
      ...batchPredictions.predictions.map((pred, index) => 
        `${index + 1},${pred},${batchPredictions.confidences[index].toFixed(3)}`
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <Target className="h-8 w-8 text-purple-600" />
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Exoplanet Predictions</h1>
          <p className="text-gray-600">Make predictions using trained ML models</p>
        </div>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">{success}</AlertDescription>
        </Alert>
      )}

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>Model Selection</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="model-select">Select Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a trained model" />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.model_type} - {(model.accuracy * 100).toFixed(1)}% accuracy
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <div className="text-sm text-gray-600">
                {availableModels.length === 0 ? (
                  <span className="text-red-600">No trained models available</span>
                ) : (
                  <span>{availableModels.length} models available</span>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="single" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="single">Single Prediction</TabsTrigger>
          <TabsTrigger value="batch">Batch Predictions</TabsTrigger>
        </TabsList>

        <TabsContent value="single">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Features */}
            <Card>
              <CardHeader>
                <CardTitle>Input Features</CardTitle>
                <CardDescription>
                  Enter exoplanet and host star properties
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 gap-4 max-h-96 overflow-y-auto">
                  {CONSTANTS.FEATURE_COLUMNS.map((feature) => (
                    <div key={feature}>
                      <Label htmlFor={feature}>
                        {CONSTANTS.FEATURE_LABELS[feature] || feature}
                      </Label>
                      <Input
                        id={feature}
                        type="number"
                        step="any"
                        placeholder="Enter value (optional)"
                        value={singleFeatures[feature]}
                        onChange={(e) => handleSingleFeatureChange(feature, e.target.value)}
                      />
                    </div>
                  ))}
                </div>
                
                <Separator />
                
                <Button
                  onClick={handleSinglePrediction}
                  disabled={loading.single || !selectedModel}
                  className="w-full"
                >
                  {loading.single ? (
                    <>
                      <Zap className="mr-2 h-4 w-4 animate-spin" />
                      Predicting...
                    </>
                  ) : (
                    <>
                      <Zap className="mr-2 h-4 w-4" />
                      Make Prediction
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Prediction Result */}
            <Card>
              <CardHeader>
                <CardTitle>Prediction Result</CardTitle>
              </CardHeader>
              <CardContent>
                {singlePrediction ? (
                  <div className="space-y-4">
                    <div className="text-center">
                      <Badge variant="outline" className="text-lg px-4 py-2">
                        {singlePrediction.prediction}
                      </Badge>
                      <p className="text-sm text-gray-600 mt-2">
                        Confidence: {utils.formatPercentage(singlePrediction.confidence)}
                      </p>
                    </div>

                    {singlePrediction.probabilities && (
                      <div className="space-y-2">
                        <h4 className="font-medium">Class Probabilities</h4>
                        {Object.entries(singlePrediction.probabilities).map(([type, prob]) => (
                          <div key={type} className="flex justify-between items-center">
                            <span className="text-sm">{type}</span>
                            <div className="flex items-center space-x-2">
                              <div className="w-24 bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-blue-600 h-2 rounded-full"
                                  style={{ width: `${prob * 100}%` }}
                                />
                              </div>
                              <span className="text-sm text-gray-600 w-12">
                                {utils.formatPercentage(prob)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {singlePrediction.feature_importance && (
                      <div className="space-y-2">
                        <h4 className="font-medium">Feature Importance</h4>
                        {Object.entries(singlePrediction.feature_importance)
                          .sort(([,a], [,b]) => b - a)
                          .slice(0, 5)
                          .map(([feature, importance]) => (
                            <div key={feature} className="flex justify-between items-center">
                              <span className="text-sm">{CONSTANTS.FEATURE_LABELS[feature] || feature}</span>
                              <span className="text-sm text-gray-600">
                                {(importance * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Enter features and click "Make Prediction" to see results</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="batch">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Batch Input */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Batch Data Input</span>
                  <Button variant="outline" size="sm" onClick={generateSampleData}>
                    <Upload className="h-4 w-4 mr-2" />
                    Sample Data
                  </Button>
                </CardTitle>
                <CardDescription>
                  Enter CSV data with feature columns
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="batch-data">CSV Data</Label>
                  <Textarea
                    id="batch-data"
                    placeholder="pl_rade,pl_masse,pl_orbper,st_teff,st_mass&#10;1.2,1.5,365.25,5778,1.0&#10;11.2,317.8,11.86,5778,1.0"
                    value={batchData}
                    onChange={(e) => setBatchData(e.target.value)}
                    rows={10}
                    className="font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    First row should contain column headers
                  </p>
                </div>

                <Button
                  onClick={handleBatchPrediction}
                  disabled={loading.batch || !selectedModel}
                  className="w-full"
                >
                  {loading.batch ? (
                    <>
                      <Zap className="mr-2 h-4 w-4 animate-spin" />
                      Processing Batch...
                    </>
                  ) : (
                    <>
                      <Zap className="mr-2 h-4 w-4" />
                      Run Batch Predictions
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Batch Results */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Batch Results</span>
                  {batchPredictions && (
                    <Button variant="outline" size="sm" onClick={exportResults}>
                      <Download className="h-4 w-4 mr-2" />
                      Export CSV
                    </Button>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {batchPredictions ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <p className="text-2xl font-bold text-blue-600">
                          {batchPredictions.predictions.length}
                        </p>
                        <p className="text-sm text-gray-600">Total Predictions</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-green-600">
                          {utils.formatPercentage(
                            batchPredictions.confidences.reduce((a, b) => a + b, 0) / 
                            batchPredictions.confidences.length
                          )}
                        </p>
                        <p className="text-sm text-gray-600">Avg Confidence</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-purple-600">
                          {new Set(batchPredictions.predictions).size}
                        </p>
                        <p className="text-sm text-gray-600">Unique Types</p>
                      </div>
                    </div>

                    <Separator />

                    <div className="max-h-64 overflow-y-auto">
                      <div className="space-y-2">
                        {batchPredictions.predictions.map((prediction, index) => (
                          <div key={index} className="flex justify-between items-center p-2 border rounded">
                            <span className="text-sm font-medium">Sample {index + 1}</span>
                            <div className="flex items-center space-x-2">
                              <Badge variant="outline">{prediction}</Badge>
                              <span className="text-xs text-gray-500">
                                {utils.formatPercentage(batchPredictions.confidences[index])}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <Upload className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Upload batch data to see prediction results</p>
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

export default Predictions;

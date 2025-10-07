import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Checkbox } from './ui/checkbox';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { mlAPI, CONSTANTS } from '../services/api';
import { Brain, Play, CheckCircle, AlertCircle, Clock, TrendingUp } from 'lucide-react';

const MLTraining = () => {
  const [trainingStatus, setTrainingStatus] = useState({ training_in_progress: false });
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState(['Random Forest', 'XGBoost']);
  const [selectedFeatures, setSelectedFeatures] = useState(CONSTANTS.FEATURE_COLUMNS.slice(0, 8));
  const [trainingConfig, setTrainingConfig] = useState({
    target_column: 'exoplanet_type',
    test_size: 0.2,
    random_state: 42,
    apply_scaling: true,
    handle_missing: 'median'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    fetchTrainingStatus();
    fetchAvailableModels();
    
    // Poll training status every 5 seconds if training is in progress
    const interval = setInterval(() => {
      if (trainingStatus.training_in_progress) {
        fetchTrainingStatus();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [trainingStatus.training_in_progress]);

  const fetchTrainingStatus = async () => {
    try {
      const status = await mlAPI.getTrainingStatus();
      setTrainingStatus(status);
    } catch (err) {
      console.error('Failed to fetch training status:', err);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const models = await mlAPI.getModels();
      setAvailableModels(models.models || []);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  const handleStartTraining = async () => {
    if (selectedAlgorithms.length === 0) {
      setError('Please select at least one algorithm');
      return;
    }
    if (selectedFeatures.length === 0) {
      setError('Please select at least one feature');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const request = {
        model_types: selectedAlgorithms,
        feature_columns: selectedFeatures,
        ...trainingConfig
      };

      await mlAPI.trainModels(request);
      setSuccess('Model training started successfully! Check the status below.');
      fetchTrainingStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAlgorithmToggle = (algorithm) => {
    setSelectedAlgorithms(prev => 
      prev.includes(algorithm) 
        ? prev.filter(a => a !== algorithm)
        : [...prev, algorithm]
    );
  };

  const handleFeatureToggle = (feature) => {
    setSelectedFeatures(prev => 
      prev.includes(feature) 
        ? prev.filter(f => f !== feature)
        : [...prev, feature]
    );
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <Brain className="h-8 w-8 text-blue-600" />
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Machine Learning Training</h1>
          <p className="text-gray-600">Train and manage exoplanet classification models</p>
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Training Configuration */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Play className="h-5 w-5" />
                <span>Training Configuration</span>
              </CardTitle>
              <CardDescription>
                Configure your machine learning training parameters
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="algorithms" className="space-y-4">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="algorithms">Algorithms</TabsTrigger>
                  <TabsTrigger value="features">Features</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                </TabsList>

                <TabsContent value="algorithms" className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium mb-3">Select ML Algorithms</h3>
                    <div className="grid grid-cols-2 gap-3">
                      {CONSTANTS.ML_ALGORITHMS.map((algorithm) => (
                        <div key={algorithm} className="flex items-center space-x-2">
                          <Checkbox
                            id={algorithm}
                            checked={selectedAlgorithms.includes(algorithm)}
                            onCheckedChange={() => handleAlgorithmToggle(algorithm)}
                          />
                          <Label htmlFor={algorithm} className="text-sm font-medium">
                            {algorithm}
                          </Label>
                        </div>
                      ))}
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      Selected: {selectedAlgorithms.length} algorithms
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="features" className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium mb-3">Select Features</h3>
                    <div className="grid grid-cols-2 gap-3">
                      {CONSTANTS.FEATURE_COLUMNS.map((feature) => (
                        <div key={feature} className="flex items-center space-x-2">
                          <Checkbox
                            id={feature}
                            checked={selectedFeatures.includes(feature)}
                            onCheckedChange={() => handleFeatureToggle(feature)}
                          />
                          <Label htmlFor={feature} className="text-sm font-medium">
                            {CONSTANTS.FEATURE_LABELS[feature] || feature}
                          </Label>
                        </div>
                      ))}
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      Selected: {selectedFeatures.length} features
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="settings" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="test_size">Test Size</Label>
                      <Select
                        value={trainingConfig.test_size.toString()}
                        onValueChange={(value) => 
                          setTrainingConfig(prev => ({ ...prev, test_size: parseFloat(value) }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0.1">10%</SelectItem>
                          <SelectItem value="0.2">20%</SelectItem>
                          <SelectItem value="0.3">30%</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label htmlFor="handle_missing">Missing Values</Label>
                      <Select
                        value={trainingConfig.handle_missing}
                        onValueChange={(value) => 
                          setTrainingConfig(prev => ({ ...prev, handle_missing: value }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="median">Fill with Median</SelectItem>
                          <SelectItem value="mean">Fill with Mean</SelectItem>
                          <SelectItem value="drop">Drop Rows</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="apply_scaling"
                      checked={trainingConfig.apply_scaling}
                      onCheckedChange={(checked) => 
                        setTrainingConfig(prev => ({ ...prev, apply_scaling: checked }))
                      }
                    />
                    <Label htmlFor="apply_scaling">Apply Feature Scaling</Label>
                  </div>
                </TabsContent>
              </Tabs>

              <Separator className="my-6" />

              <Button
                onClick={handleStartTraining}
                disabled={loading || trainingStatus.training_in_progress}
                className="w-full"
                size="lg"
              >
                {loading ? (
                  <>
                    <Clock className="mr-2 h-4 w-4 animate-spin" />
                    Starting Training...
                  </>
                ) : trainingStatus.training_in_progress ? (
                  <>
                    <Clock className="mr-2 h-4 w-4 animate-pulse" />
                    Training in Progress...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Start Training
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Training Status & Models */}
        <div className="space-y-6">
          {/* Training Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="h-5 w-5" />
                <span>Training Status</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Status</span>
                  <Badge variant={trainingStatus.training_in_progress ? "default" : "secondary"}>
                    {trainingStatus.training_in_progress ? "Training" : "Idle"}
                  </Badge>
                </div>
                
                {trainingStatus.training_in_progress && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Progress</span>
                      <span>Processing...</span>
                    </div>
                    <Progress value={75} className="h-2" />
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Available Models */}
          <Card>
            <CardHeader>
              <CardTitle>Available Models</CardTitle>
              <CardDescription>
                {availableModels.length} trained models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {availableModels.length === 0 ? (
                  <p className="text-sm text-gray-500 text-center py-4">
                    No models trained yet
                  </p>
                ) : (
                  availableModels.map((model) => (
                    <div key={model.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium text-sm">{model.model_type}</p>
                        <p className="text-xs text-gray-500">
                          Accuracy: {(model.accuracy * 100).toFixed(1)}%
                        </p>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {model.training_data_count} samples
                      </Badge>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default MLTraining;

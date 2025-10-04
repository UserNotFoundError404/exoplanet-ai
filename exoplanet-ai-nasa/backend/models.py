from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uuid

# Base models for NASA data
class ExoplanetData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pl_name: Optional[str] = Field(None, description="Planet name")
    hostname: Optional[str] = Field(None, description="Host star name")
    pl_letter: Optional[str] = Field(None, description="Planet letter")
    pl_rade: Optional[float] = Field(None, description="Planet radius (Earth radii)")
    pl_masse: Optional[float] = Field(None, description="Planet mass (Earth masses)")
    pl_orbper: Optional[float] = Field(None, description="Orbital period (days)")
    pl_orbsmax: Optional[float] = Field(None, description="Semi-major axis (AU)")
    pl_orbeccen: Optional[float] = Field(None, description="Eccentricity")
    pl_eqt: Optional[float] = Field(None, description="Equilibrium temperature (K)")
    st_rad: Optional[float] = Field(None, description="Stellar radius (Solar radii)")
    st_mass: Optional[float] = Field(None, description="Stellar mass (Solar masses)")
    st_teff: Optional[float] = Field(None, description="Stellar effective temperature (K)")
    st_met: Optional[float] = Field(None, description="Stellar metallicity")
    st_logg: Optional[float] = Field(None, description="Stellar surface gravity")
    sy_dist: Optional[float] = Field(None, description="System distance (parsecs)")
    disc_year: Optional[int] = Field(None, description="Discovery year")
    disc_facility: Optional[str] = Field(None, description="Discovery facility")
    disc_telescope: Optional[str] = Field(None, description="Discovery telescope")
    disc_instrument: Optional[str] = Field(None, description="Discovery instrument")
    data_source: str = Field(description="Source of data (Kepler, TESS, etc.)")
    exoplanet_type: Optional[str] = Field(None, description="Classified exoplanet type")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class LightCurveData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_name: str = Field(description="Target identifier")
    time: List[float] = Field(description="Time array")
    flux: List[float] = Field(description="Flux measurements")
    flux_err: Optional[List[float]] = Field(None, description="Flux uncertainties")
    quality: Optional[List[int]] = Field(None, description="Quality flags")
    mission: str = Field(description="Mission (Kepler/TESS)")
    sector: Optional[int] = Field(None, description="TESS sector or Kepler quarter")
    cadence: str = Field(description="Observation cadence")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TransitData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_name: str = Field(description="Target identifier")
    period: float = Field(description="Transit period (days)")
    duration: float = Field(description="Transit duration (hours)")
    depth: float = Field(description="Transit depth (ppm)")
    epoch: float = Field(description="Transit epoch (BJD)")
    impact_parameter: Optional[float] = Field(None, description="Impact parameter")
    planet_radius: Optional[float] = Field(None, description="Planet radius (Earth radii)")
    semi_major_axis: Optional[float] = Field(None, description="Semi-major axis (AU)")
    eccentricity: Optional[float] = Field(None, description="Orbital eccentricity")
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ML Model related models
class MLModelMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Model name")
    model_type: str = Field(description="Algorithm type (RandomForest, XGBoost, etc.)")
    version: str = Field(default="1.0")
    training_data_count: int = Field(description="Number of training samples")
    features: List[str] = Field(description="Feature columns used")
    target_classes: List[str] = Field(description="Classification target classes")
    performance_metrics: Dict[str, float] = Field(description="Model performance metrics")
    model_file_path: Optional[str] = Field(None, description="Path to saved model file")
    scaler_file_path: Optional[str] = Field(None, description="Path to saved scaler file")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

class ModelTrainingRequest(BaseModel):
    model_types: List[str] = Field(description="List of model types to train")
    feature_columns: List[str] = Field(description="Features to use for training")
    target_column: str = Field(default="exoplanet_type")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    apply_scaling: bool = Field(default=True)
    handle_missing: str = Field(default="median", pattern="^(drop|median|mean)$")

class ModelPredictionRequest(BaseModel):
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    model_type: Optional[str] = Field(None, description="Model type to use")
    features: Dict[str, float] = Field(description="Feature values for prediction")
    use_ensemble: bool = Field(default=False)

class BatchPredictionRequest(BaseModel):
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    model_type: Optional[str] = Field(None, description="Model type to use")
    data: List[Dict[str, float]] = Field(description="List of feature sets")
    use_ensemble: bool = Field(default=False)

class PredictionResponse(BaseModel):
    prediction: str = Field(description="Predicted class")
    confidence: float = Field(description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    model_used: str = Field(description="Model ID or type used")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")

class BatchPredictionResponse(BaseModel):
    predictions: List[str] = Field(description="Predicted classes")
    confidences: List[float] = Field(description="Prediction confidences")
    model_used: str = Field(description="Model ID or type used")
    ensemble_info: Optional[Dict[str, Any]] = Field(None, description="Ensemble details")

# Data loading and analysis models
class DataLoadRequest(BaseModel):
    data_sources: List[str] = Field(description="Data sources to load")
    limit_per_source: int = Field(default=2000, ge=100, le=10000)
    include_light_curves: bool = Field(default=False)
    include_transit_data: bool = Field(default=False)

class DataLoadResponse(BaseModel):
    total_records: int = Field(description="Total records loaded")
    sources_loaded: List[str] = Field(description="Successfully loaded sources")
    records_per_source: Dict[str, int] = Field(description="Records count by source")
    features_count: int = Field(description="Number of feature columns")
    load_time: float = Field(description="Load time in seconds")

class AnalysisRequest(BaseModel):
    target_name: Optional[str] = Field(None, description="Specific target to analyze")
    analysis_type: str = Field(description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")

class VisualizationRequest(BaseModel):
    chart_type: str = Field(description="Type of visualization")
    target_name: Optional[str] = Field(None, description="Specific target")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Chart parameters")

# System models
class StatusResponse(BaseModel):
    status: str = "operational"
    database_connected: bool = True
    models_available: int = 0
    data_records: int = 0
    last_data_load: Optional[datetime] = None
    system_info: Dict[str, Any] = Field(default_factory=dict)

# Classification target types
EXOPLANET_TYPES = [
    "Hot Jupiter",
    "Warm Jupiter", 
    "Cold Jupiter",
    "Super Earth",
    "Sub Neptune",
    "Neptune-like",
    "Terrestrial",
    "Mini Neptune",
    "Gas Giant",
    "Rocky Planet"
]

# Available data sources
DATA_SOURCES = [
    "Kepler Confirmed Planets",
    "Kepler KOI Cumulative", 
    "TESS Objects of Interest",
    "Planetary Systems Composite",
    "NASA Exoplanet Archive"
]

# Supported ML algorithms
ML_ALGORITHMS = [
    "Random Forest",
    "XGBoost", 
    "SVM",
    "Logistic Regression",
    "Neural Network",
    "Extra Trees",
    "Gradient Boosting"
]
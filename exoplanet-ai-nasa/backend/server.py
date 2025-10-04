from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import uuid
from datetime import datetime
import asyncio
import pandas as pd
import numpy as np

# Import our custom modules
from models import (
    StatusResponse, DataLoadRequest, DataLoadResponse, ModelTrainingRequest, 
    ModelPredictionRequest, BatchPredictionRequest, PredictionResponse, 
    BatchPredictionResponse, AnalysisRequest, VisualizationRequest,
    ExoplanetData, LightCurveData, TransitData, MLModelMetadata,
    DATA_SOURCES, ML_ALGORITHMS, EXOPLANET_TYPES
)
from nasa_data_loader import NASADataLoader
from ml_engine import MLEngine  
from visualization_engine import VisualizationEngine

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Collections
exoplanets_collection = db.exoplanets
light_curves_collection = db.light_curves
transits_collection = db.transits
models_collection = db.ml_models

# Initialize engines
data_loader = NASADataLoader()
ml_engine = MLEngine(models_collection)
viz_engine = VisualizationEngine()

# Create the main app
app = FastAPI(
    title="NASA Stellar Data Analysis & ML Platform",
    description="Comprehensive platform for NASA exoplanet data analysis, machine learning, and visualization",
    version="1.0.0"
)

# Create API router
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
current_dataset = None
training_in_progress = False

@api_router.get("/", response_model=StatusResponse)
async def get_system_status():
    """Get system status and information."""
    try:
        # Count records
        exoplanet_count = await exoplanets_collection.count_documents({})
        model_count = await models_collection.count_documents({'is_active': True})
        
        # Get last data load
        last_load = await exoplanets_collection.find_one(
            {}, 
            sort=[('created_at', -1)]
        )
        
        system_info = {
            'nasa_api_configured': bool(os.environ.get('NASA_API_KEY')),
            'available_algorithms': ML_ALGORITHMS,
            'supported_data_sources': DATA_SOURCES,
            'exoplanet_types': EXOPLANET_TYPES
        }
        
        return StatusResponse(
            database_connected=True,
            models_available=model_count,
            data_records=exoplanet_count,
            last_data_load=last_load['created_at'] if last_load else None,
            system_info=system_info
        )
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return StatusResponse(
            status="error",
            database_connected=False,
            system_info={"error": str(e)}
        )

# Data Loading Endpoints
@api_router.post("/data/load", response_model=DataLoadResponse)
async def load_nasa_data(request: DataLoadRequest, background_tasks: BackgroundTasks):
    """Load data from NASA sources."""
    global current_dataset
    
    try:
        logger.info(f"Starting data load for sources: {request.data_sources}")
        start_time = datetime.utcnow()
        
        datasets = []
        sources_loaded = []
        records_per_source = {}
        
        for source in request.data_sources:
            try:
                if source == "Kepler Confirmed Planets":
                    df = await data_loader.load_kepler_confirmed_planets(limit=request.limit_per_source)
                elif source == "Kepler KOI Cumulative":
                    df = await data_loader.load_kepler_koi_cumulative(limit=request.limit_per_source)
                elif source == "TESS Objects of Interest":
                    df = await data_loader.load_tess_toi(limit=request.limit_per_source)
                elif source == "Planetary Systems Composite":
                    df = await data_loader.load_planetary_systems(limit=request.limit_per_source)
                else:
                    logger.warning(f"Unknown data source: {source}")
                    continue
                
                if df is not None and not df.empty:
                    datasets.append(df)
                    sources_loaded.append(source)
                    records_per_source[source] = len(df)
                    logger.info(f"Loaded {len(df)} records from {source}")
                
            except Exception as e:
                logger.error(f"Error loading {source}: {str(e)}")
                continue
        
        if not datasets:
            raise HTTPException(status_code=400, detail="Failed to load data from any source")
        
        # Combine datasets
        combined_df = data_loader.combine_datasets(datasets)
        current_dataset = combined_df
        
        # Store in database (background task)
        background_tasks.add_task(store_exoplanet_data, combined_df)
        
        # Load additional data if requested
        if request.include_light_curves or request.include_transit_data:
            background_tasks.add_task(load_additional_data, 
                                    combined_df['pl_name'].head(10).tolist(), 
                                    request.include_light_curves, 
                                    request.include_transit_data)
        
        load_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DataLoadResponse(
            total_records=len(combined_df),
            sources_loaded=sources_loaded,
            records_per_source=records_per_source,
            features_count=len(combined_df.columns),
            load_time=load_time
        )
        
    except Exception as e:
        logger.error(f"Error in data loading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/overview")
async def get_data_overview():
    """Get overview of loaded data."""
    global current_dataset
    
    try:
        if current_dataset is None:
            # Try to load from database
            cursor = exoplanets_collection.find().limit(10000)
            records = await cursor.to_list(length=10000)
            if not records:
                raise HTTPException(status_code=404, detail="No data loaded. Please load data first.")
            
            current_dataset = pd.DataFrame([{k: v for k, v in record.items() if k != '_id'} for record in records])
        
        # Generate overview visualization
        overview_chart = viz_engine.create_exoplanet_overview(current_dataset)
        
        return {
            "total_records": len(current_dataset),
            "features": list(current_dataset.columns),
            "data_sources": current_dataset['data_source'].value_counts().to_dict() if 'data_source' in current_dataset.columns else {},
            "planet_types": current_dataset['exoplanet_type'].value_counts().to_dict() if 'exoplanet_type' in current_dataset.columns else {},
            "visualization": overview_chart
        }
    except Exception as e:
        logger.error(f"Error getting data overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Machine Learning Endpoints
@api_router.post("/ml/train")
async def train_models(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train machine learning models."""
    global training_in_progress
    
    if training_in_progress:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    try:
        if current_dataset is None or len(current_dataset) == 0:
            raise HTTPException(status_code=400, detail="No data available for training. Please load data first.")
        
        training_in_progress = True
        
        # Start training in background
        background_tasks.add_task(
            train_models_background,
            current_dataset.copy(),
            request.model_types,
            request.feature_columns,
            request.target_column,
            request.test_size,
            request.random_state,
            request.apply_scaling,
            request.handle_missing
        )
        
        return {"message": "Model training started", "status": "in_progress"}
        
    except Exception as e:
        training_in_progress = False
        logger.error(f"Error starting model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/ml/training-status")
async def get_training_status():
    """Get training status."""
    return {"training_in_progress": training_in_progress}

@api_router.get("/ml/models")
async def get_available_models():
    """Get list of available trained models."""
    try:
        models = await models_collection.find({"is_active": True}).to_list(length=100)
        
        model_list = []
        for model in models:
            model_info = {
                "id": model["id"],
                "name": model["name"],
                "model_type": model["model_type"],
                "accuracy": model["performance_metrics"]["accuracy"],
                "created_at": model["created_at"],
                "training_data_count": model["training_data_count"]
            }
            model_list.append(model_info)
        
        return {"models": model_list, "total": len(model_list)}
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/ml/predict", response_model=PredictionResponse)
async def predict_single(request: ModelPredictionRequest):
    """Make prediction for a single exoplanet."""
    try:
        result = await ml_engine.predict_single(
            features=request.features,
            model_id=request.model_id,
            model_type=request.model_type
        )
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error making single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/ml/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple exoplanets."""
    try:
        result = await ml_engine.predict_batch(
            features_list=request.data,
            model_id=request.model_id,
            model_type=request.model_type,
            use_ensemble=request.use_ensemble
        )
        return BatchPredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/ml/performance")
async def get_model_performance():
    """Get performance metrics for all models."""
    try:
        performance_data = await ml_engine.get_model_performance()
        
        # Create performance chart
        if performance_data:
            performance_chart = viz_engine.create_model_performance_chart(performance_data)
        else:
            performance_chart = {"type": "error", "message": "No models available"}
        
        return {
            "models": performance_data,
            "visualization": performance_chart
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Visualization Endpoints
@api_router.post("/visualizations/create")
async def create_visualization(request: VisualizationRequest):
    """Create custom visualizations."""
    try:
        if current_dataset is None:
            raise HTTPException(status_code=404, detail="No data available for visualization")
        
        chart_type = request.chart_type
        parameters = request.parameters
        
        if chart_type == "correlation_heatmap":
            result = viz_engine.create_correlation_heatmap(current_dataset)
        elif chart_type == "discovery_timeline":
            result = viz_engine.create_discovery_timeline(current_dataset)
        elif chart_type == "planet_classification":
            x_feature = parameters.get('x_feature', 'pl_rade')
            y_feature = parameters.get('y_feature', 'pl_masse')
            result = viz_engine.create_planet_classification_plot(current_dataset, x_feature, y_feature)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported chart type: {chart_type}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/visualizations/light-curve/{target_name}")
async def get_light_curve(target_name: str, mission: str = "TESS"):
    """Get light curve visualization for a specific target."""
    try:
        # Load light curve data
        light_curve_data = await data_loader.load_light_curve_data(target_name, mission)
        
        if not light_curve_data:
            raise HTTPException(status_code=404, detail=f"Light curve data not found for {target_name}")
        
        # Create visualization
        chart = viz_engine.create_light_curve(
            time=light_curve_data['time'],
            flux=light_curve_data['flux'],
            flux_err=light_curve_data.get('flux_err'),
            target_name=target_name,
            mission=mission
        )
        
        return chart
        
    except Exception as e:
        logger.error(f"Error getting light curve for {target_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/visualizations/transit/{target_name}")
async def get_transit_analysis(target_name: str):
    """Get transit analysis visualization for a specific target."""
    try:
        # Load light curve and transit data
        light_curve_data = await data_loader.load_light_curve_data(target_name)
        transit_data = await data_loader.get_transit_data(target_name)
        
        if not light_curve_data or not transit_data:
            raise HTTPException(status_code=404, detail=f"Transit data not found for {target_name}")
        
        # Create visualization
        chart = viz_engine.create_transit_analysis(light_curve_data, transit_data)
        
        return chart
        
    except Exception as e:
        logger.error(f"Error getting transit analysis for {target_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility functions
async def store_exoplanet_data(df: pd.DataFrame):
    """Store exoplanet data in database (background task)."""
    try:
        # Convert DataFrame to list of dicts
        records = df.to_dict('records')
        
        # Add timestamps and IDs
        for record in records:
            record['id'] = str(uuid.uuid4())
            record['created_at'] = datetime.utcnow()
            # Convert numpy types to Python types
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = value.item()
                elif pd.isna(value):
                    record[key] = None
        
        # Clear existing data and insert new
        await exoplanets_collection.delete_many({})
        if records:
            await exoplanets_collection.insert_many(records)
        
        logger.info(f"Stored {len(records)} exoplanet records in database")
        
    except Exception as e:
        logger.error(f"Error storing exoplanet data: {str(e)}")

async def load_additional_data(target_names: List[str], include_light_curves: bool, include_transit_data: bool):
    """Load additional data (light curves, transit data) for targets."""
    try:
        for target_name in target_names:
            if include_light_curves:
                light_curve_data = await data_loader.load_light_curve_data(target_name)
                if light_curve_data:
                    light_curve_data['id'] = str(uuid.uuid4())
                    light_curve_data['created_at'] = datetime.utcnow()
                    await light_curves_collection.insert_one(light_curve_data)
            
            if include_transit_data:
                transit_data = await data_loader.get_transit_data(target_name)
                if transit_data:
                    transit_data['id'] = str(uuid.uuid4())
                    transit_data['created_at'] = datetime.utcnow()
                    await transits_collection.insert_one(transit_data)
        
        logger.info(f"Loaded additional data for {len(target_names)} targets")
        
    except Exception as e:
        logger.error(f"Error loading additional data: {str(e)}")

async def train_models_background(df: pd.DataFrame, model_types: List[str], feature_columns: List[str], 
                                target_column: str, test_size: float, random_state: int, 
                                apply_scaling: bool, handle_missing: str):
    """Train models in background."""
    global training_in_progress
    
    try:
        logger.info(f"Starting background training for models: {model_types}")
        
        # Prepare data
        X, y, preprocessing_info = await ml_engine.prepare_training_data(
            data=df,
            feature_columns=feature_columns,
            target_column=target_column,
            handle_missing=handle_missing,
            apply_scaling=apply_scaling
        )
        
        # Train each model
        results = []
        for model_type in model_types:
            try:
                result = await ml_engine.train_model(
                    X=X,
                    y=y,
                    model_type=model_type,
                    preprocessing_info=preprocessing_info,
                    test_size=test_size,
                    random_state=random_state
                )
                results.append(result)
                logger.info(f"Successfully trained {model_type} model")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                continue
        
        logger.info(f"Background training completed. Trained {len(results)} models successfully.")
        
    except Exception as e:
        logger.error(f"Error in background training: {str(e)}")
    finally:
        training_in_progress = False

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

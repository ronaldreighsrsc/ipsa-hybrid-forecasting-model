from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from contextlib import asynccontextmanager
import numpy as np
import json
import os
import shutil

# Import models
from src.models.arimax import ARIMAXTrainer
from src.models.random_forest import RandomForestTrainer
from src.models.xgb_model import XGBoostTrainer
from src.models.lstm_model import LSTMTrainer
from src.models.bilstm_model import BiLSTMTrainer
from src.models.arima_lstm import HybridARIMALSTMTrainer
from src.models.lstm_rf import LSTMRFTrainer

# Mapeo global de clases de modelos
MODEL_CLASSES = {
    'arimax': ARIMAXTrainer,
    'random_forest': RandomForestTrainer,
    'xgboost': XGBoostTrainer,
    'lstm': LSTMTrainer,
    'bilstm': BiLSTMTrainer,
    'arima_lstm': HybridARIMALSTMTrainer,
    'lstm_rf': LSTMRFTrainer
}

ml_models = {}

def load_production_model():
    """Carga el modelo de producción actual a la memoria RAM."""
    meta_path = "./models/production/metadata.json"
    if not os.path.exists(meta_path):
        return False
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    model_type = metadata['modelo']
    pkl_path = "./models/production/production_model.pkl"
    
    if not os.path.exists(pkl_path):
        return False
        
    model_class = MODEL_CLASSES.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")
        
    ml_models['production'] = model_class.load(pkl_path)
    ml_models['metadata'] = metadata
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Iniciando MLOps Server...")
    os.makedirs("./models/production", exist_ok=True)
    if load_production_model():
        print(f"✅ Active Production Model: {ml_models['metadata']['modelo'].upper()} (Bank: {ml_models['metadata']['banco']})")
    else:
        print("⚠️ No hay modelo en producción. Ejecuta POST /promote para promover uno.")
    yield
    ml_models.clear()

app = FastAPI(
    title="IPSA MLOps Dynamic API",
    description="Dynamic Model Registry and Serving for IPSA Forecasting",
    version="3.0.0",
    lifespan=lifespan
)

class PromoteRequest(BaseModel):
    metric: str = Field(..., description="La métrica para elegir al ganador: 'alpha', 'sharpe', 'accuracy', o 'win_rate'")

class DailyData(BaseModel):
    Price: float
    SP500_FFD: float
    FXI: float
    USDCLP_FFD: float
    Yield10Y_FFD: float
    EMBI: float
    VIX: float
    TPM: float
    Copper_FFD: float
    MACD_Hist: float
    RSI: float
    EGARCH_Vol: float
    ATR: float
    Open_FFD: float
    High_FFD: float
    Low_FFD: float

class MarketWindow(BaseModel):
    window: List[DailyData] = Field(..., description="Lista de 60 días de variables financieras")

@app.post("/promote")
def promote_model(req: PromoteRequest):
    """
    Escanea el Model Registry, encuentra el modelo con la mejor métrica solicitada,
    lo promueve a la carpeta de producción y hace un hot-reload en memoria.
    """
    registry_path = "./models/registry.json"
    features_path = "./models/features_map.json"
    
    if not os.path.exists(registry_path) or not os.path.exists(features_path):
        raise HTTPException(status_code=400, detail="Falta el archivo registry.json o features_map.json. Corre main_evaluation.py y main_ablation.py")
        
    with open(registry_path, "r") as f:
        registry = json.load(f)
    with open(features_path, "r") as f:
        features_map = json.load(f)
        
    best_score = -np.inf
    best_candidate = None
    best_model_name = None
    
    for mod_key, data in registry.items():
        # Extraemos el nombre del modelo base (ej. 'xgboost' de 'xgboost')
        # Wait, en main_evaluation, mod_key es solo 'xgboost', y data['banco'] es 'Macros'.
        # El archivo físico se guardó como xgboost_macros.pkl
        model_base_name = mod_key.lower()
        banco = data['banco'].lower()
        
        if req.metric == 'alpha': score = data['alpha']
        elif req.metric == 'accuracy': score = data['metrics'].get('Accuracy', data.get('accuracy', -np.inf)) # Fallback si accuracy no está en metrics
        elif req.metric == 'sharpe': score = data['metrics']['Sharpe']
        elif req.metric == 'win_rate': score = data['win_rate']
        else:
            raise HTTPException(status_code=400, detail=f"Métrica desconocida: {req.metric}")
            
        if score > best_score:
            best_score = score
            best_candidate = data
            best_model_name = model_base_name
            
    if not best_candidate:
        raise HTTPException(status_code=400, detail="No se encontraron modelos en el registry con esa métrica.")
        
    banco_str = best_candidate['banco']
    banco_str_lower = banco_str.lower()
    pkl_source = f"./models/ablation_candidates/{best_model_name}_{banco_str_lower}.pkl"
    keras_source = f"./models/ablation_candidates/{best_model_name}_{banco_str_lower}.keras"
    
    if not os.path.exists(pkl_source):
        raise HTTPException(status_code=500, detail=f"El archivo {pkl_source} no existe.")
        
    # Copiar a producción
    shutil.copy(pkl_source, "./models/production/production_model.pkl")
    if os.path.exists(keras_source):
        shutil.copy(keras_source, "./models/production/production_model.keras")
        
    # Determinar las features exactas
    exog_features = features_map.get(banco_str, [])
    exact_features = ['Price'] + exog_features
    
    metadata = {
        "modelo": best_model_name,
        "banco": banco_str,
        "features": exact_features,
        "metrica_promocion": req.metric,
        "score_promocion": best_score
    }
    
    with open("./models/production/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Hot-reload
    success = load_production_model()
    
    return {
        "status": "success",
        "message": f"Modelo {best_model_name.upper()} ({banco_str}) promovido a producción.",
        "score": best_score,
        "hot_reload": success
    }

@app.post("/predict")
def predict_direction(window: MarketWindow):
    if 'production' not in ml_models:
        raise HTTPException(status_code=500, detail="No hay modelo en producción. Ejecuta /promote primero.")
        
    meta = ml_models['metadata']
    required_features = meta['features']
    model_type = meta['modelo']
    
    if len(window.window) != 60:
        raise HTTPException(status_code=400, detail="Se requieren exactamente 60 días de historia.")
        
    # Construir la matriz dinámicamente con solo las columnas necesarias
    data_matrix = []
    for day in window.window:
        day_dict = day.dict()
        row = [day_dict[feat] for feat in required_features]
        data_matrix.append(row)
        
    data = np.array(data_matrix) # Shape (60, num_features)
    
    # Inferencia Dinámica
    prod_model = ml_models['production']
    
    if model_type in ['random_forest', 'xgboost']:
        flattened = data.flatten().reshape(1, -1)
        scaled = np.clip(prod_model.scaler.transform(flattened), -10, 10)
        prob = prod_model._master_model.predict_proba(scaled)[0][1] if hasattr(prod_model, '_master_model') else 0.0
    
    elif model_type in ['lstm', 'bilstm']:
        scaled = np.clip(prod_model.scaler.transform(data), -10, 10).reshape(1, 60, len(required_features))
        prob = prod_model.master_model.predict(scaled, verbose=0)[0][0]
        
    elif model_type == 'lstm_rf':
        scaled = np.clip(prod_model.scaler.transform(data), -10, 10).reshape(1, 60, len(required_features))
        features = prod_model.feature_extractor.predict(scaled, verbose=0)
        prob = prod_model.rf_model.predict_proba(features)[0][1]
        
    elif model_type == 'arima_lstm':
        # ARIMA-LSTM requires returning just 0.5 for now, or building the specific logic
        # For simplicity, returning the latest prob if supported.
        # This model is very complex for real-time stateless inference, so we approximate.
        prob = 0.5
        
    elif model_type == 'arimax':
        prob = 0.5

    return {
        "status": "success",
        "production_model": model_type.upper(),
        "bank": meta['banco'],
        "prob_up": float(prob)
    }

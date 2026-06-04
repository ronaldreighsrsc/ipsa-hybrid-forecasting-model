import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Importación de todos los motores predictivos
from models.arimax import ARIMAXTrainer
from models.random_forest import RandomForestTrainer
from models.xgb_model import XGBoostTrainer
from models.lstm_model import LSTMTrainer
from models.bilstm_model import BiLSTMTrainer
from models.arima_lstm import HybridARIMALSTMTrainer
from models.lstm_rf import LSTMRFTrainer

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==============================================================================
# Modelos disponibles: 'ARIMAX', 'RANDOM_FOREST', 'XGBOOST', 'LSTM', 'BILSTM', 'ARIMA_LSTM', 'LSTM_RF'
MODELOS_A_CORRER = [
    #'ARIMAX', 
    #'RANDOM_FOREST', 
    #'XGBOOST', 
    'LSTM'
    #'BILSTM', 
    #'ARIMA_LSTM', 
    #'LSTM_RF'
]

# Configura aquí el nombre de la variable a predecir
# Si en ipsa_master_processed.csv está como 'Price_FFD', usa 'Price_FFD'
TARGET_COL = 'Price_FFD' 

# ==============================================================================
# GRILLAS DE HIPERPARÁMETROS (GRIDS)
# ==============================================================================
# Random Forest
RF_GRID = [
    {'n_estimators': 100, 'max_depth': 5},
    {'n_estimators': 150, 'max_depth': 7}
]

# XGBoost
XGB_GRID = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
    {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.01, 'subsample': 0.8}
]

# LSTM / BiLSTM / LSTM_RF
NN_GRID = [
    {'units': 50, 'dropout': 0.2},
    {'units': 100, 'dropout': 0.3}
]
# Nota: ARIMAX y ARIMA_LSTM hacen su propia búsqueda por AIC internamente.

# ==============================================================================
# BANCOS DE VARIABLES (ESTUDIO DE ABLACIÓN)
# ==============================================================================
bancos = {
    "Univariado": [],
    "Macros": ['TPM', 'EMBI', 'Copper_FFD', 'Yield10Y_FFD'],
    "Tecnicos": ['MACD_Hist', 'RSI', 'ATR', 'EGARCH_Vol'],
    "Precio_Puro": ['Open_FFD', 'High_FFD', 'Low_FFD'],
    "Hibrido_Precio_Tec": ['Open_FFD', 'High_FFD', 'Low_FFD', 'EGARCH_Vol', 'RSI'],
    "Global": ['SP500_FFD', 'VIX', 'FXI'],
    "Kitchen_Sink_Total": ['SP500_FFD', 'FXI', 'USDCLP_FFD', 'Yield10Y_FFD', 
                           'EMBI', 'VIX', 'TPM', 'Copper_FFD', 'MACD_Hist', 
                           'RSI', 'EGARCH_Vol', 'ATR']
}

def run_ablation():
    print("🚀 Iniciando Torneo de Modelos (Estudio de Ablación)...")

    # 1. Carga de datos
    data_path = "./data/processed/ipsa_master_processed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ No se encontró {data_path}.")
        
    df_base = pd.read_csv(data_path)
    df_base['Date'] = pd.to_datetime(df_base['Date'])
    df_base = df_base.sort_values('Date').set_index('Date')
    
    # Validar que exista el Target
    global TARGET_COL
    if TARGET_COL not in df_base.columns:
        print(f"⚠️ TARGET_COL '{TARGET_COL}' no encontrado. Columnas disponibles: {list(df_base.columns)}")
        print("Intentando usar 'Price' como fallback...")
        TARGET_COL = 'Price'

    os.makedirs("./src/evaluation/results", exist_ok=True)
    resultados_globales = []

    # 2. Iteración de Modelos
    for nombre_modelo in MODELOS_A_CORRER:
        print(f"\n{'#'*70}\n🏆 EVALUANDO MODELO: {nombre_modelo}\n{'#'*70}")
        
        # Instanciar el Trainer correspondiente
        if nombre_modelo == 'ARIMAX':
            trainer = ARIMAXTrainer(p_max=2, q_max=2, retrain_step=50)
        elif nombre_modelo == 'RANDOM_FOREST':
            trainer = RandomForestTrainer(look_back=60, retrain_step=50)
        elif nombre_modelo == 'XGBOOST':
            trainer = XGBoostTrainer(look_back=60, retrain_step=50)
        elif nombre_modelo == 'LSTM':
            trainer = LSTMTrainer(look_back=60, retrain_step=50)
        elif nombre_modelo == 'BILSTM':
            trainer = BiLSTMTrainer(look_back=60, retrain_step=50)
        elif nombre_modelo == 'ARIMA_LSTM':
            trainer = HybridARIMALSTMTrainer(look_back=60, retrain_step=50)
        elif nombre_modelo == 'LSTM_RF':
            trainer = LSTMRFTrainer(look_back=60, retrain_step=50)
        else:
            print(f"❌ Modelo no reconocido: {nombre_modelo}")
            continue

        # 3. Iteración de Bancos
        for nombre_banco, features in bancos.items():
            print(f"\n{'-'*50}\n🧠 BANCO: {nombre_banco} | MODELO: {nombre_modelo}\n{'-'*50}")
            
            valid_features = [c for c in features if c in df_base.columns]
            
            # --- RUTAS DE EJECUCIÓN SEGÚN INTERFAZ DEL MODELO ---
            
            if nombre_modelo in ['ARIMAX', 'ARIMA_LSTM']:
                # Interfaz Pandas (Series de Tiempo)
                if valid_features:
                    df_exog_lagged = df_base[valid_features].shift(1)
                    df_model = pd.concat([df_base[TARGET_COL], df_exog_lagged], axis=1).dropna()
                    exog = df_model[valid_features]
                else:
                    df_model = df_base[[TARGET_COL]].dropna()
                    exog = pd.DataFrame()
                    
                target = df_model[TARGET_COL]
                train_size = int(len(target) * 0.8)
                
                if nombre_modelo == 'ARIMAX':
                    best_params = trainer.find_best_order(target.iloc[:train_size], exog.iloc[:train_size])
                    pred_probs = trainer.walk_forward_predict(target, exog, train_size, best_params)
                else: # ARIMA_LSTM
                    best_params = trainer.find_best_params(target.iloc[:train_size], exog.iloc[:train_size], NN_GRID)
                    pred_probs, _ = trainer.walk_forward_predict(target, exog, train_size, best_params)
                    
                # Evaluar
                test_target = target.iloc[train_size:]
                train_target = target.iloc[:train_size]
                y_real_bin = (test_target.values > np.roll(test_target.values, 1)).astype(int)
                y_real_bin[0] = (test_target.iloc[0] > train_target.iloc[-1]).astype(int)
                final_acc = accuracy_score(y_real_bin, (np.array(pred_probs) > 0.5).astype(int))

            else:
                # Interfaz Machine Learning / Deep Learning (Secuencias 2D/3D)
                # Seleccionar la grilla correcta
                if nombre_modelo == 'RANDOM_FOREST':
                    grid = RF_GRID
                elif nombre_modelo == 'XGBOOST':
                    grid = XGB_GRID
                else:
                    grid = NN_GRID
                    
                X, y = trainer.prepare_sequences(df_base, TARGET_COL, valid_features)
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                best_params = trainer.find_best_params(X_train, y_train, grid)
                pred_probs, importances = trainer.walk_forward_predict(X_train, y_train, X_test, y_test, best_params)
                
                final_acc = accuracy_score(y_test, (np.array(pred_probs) > 0.5).astype(int))
                
                if importances is not None and nombre_modelo in ['RANDOM_FOREST', 'XGBOOST']:
                    # Opcional: imprimir el top 3 de features
                    top_indices = np.argsort(importances)[::-1][:3]
                    print("   Top 3 Variables más importantes:")
                    for idx in top_indices:
                        if idx == 0:
                            print(f"    - Autoregresivo ({TARGET_COL}): {importances[idx]:.2%}")
                        elif (idx - 1) < len(valid_features):
                            print(f"    - {valid_features[idx - 1]}: {importances[idx]:.2%}")

            print(f"✅ Exactitud Final OOS ({nombre_modelo} - {nombre_banco}): {final_acc:.2%}")
            
            # Guardar resultados
            resultados_globales.append({
                "Modelo": nombre_modelo,
                "Banco": nombre_banco,
                "Mejores_Params": str(best_params),
                "Accuracy_Test": final_acc
            })
            
            # Guardar NumPy Arrays (Probabilidades)
            npy_path = f'./src/evaluation/results/probs_{nombre_modelo.lower()}_{nombre_banco}.npy'
            np.save(npy_path, np.array(pred_probs))

    # 4. Resumen Global
    if resultados_globales:
        df_res = pd.DataFrame(resultados_globales)
        print("\n" + "="*60 + "\n🏆 TABLA DE ABLACIÓN GLOBAL 🏆\n" + "="*60)
        print(df_res.to_string(index=False))
        df_res.to_csv('./src/evaluation/results/tabla_ablacion_global.csv', index=False)
        print("\nResultados consolidados guardados en: ./src/evaluation/results/tabla_ablacion_global.csv")

if __name__ == "__main__":
    run_ablation()
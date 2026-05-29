import os
import warnings
from preprocessing.data_loader import DataLoader
from preprocessing.technical_features import TechnicalFeatureEngineer
from preprocessing.volatility import VolatilityModeler
from preprocessing.stationarity import FractionalDifferencer

warnings.filterwarnings("ignore")

def run_preprocessing_pipeline():
    print("🚀 Iniciando Pipeline de Preprocesamiento de Datos (Bloque 1)...")

    print("\n--- PASO 1: Extracción y Consolidación ---")
    loader = DataLoader(raw_data_path="./data/raw/")
    df = loader.load_and_merge()

    print("\n--- PASO 2: Ingeniería de Features Técnicos ---")
    # Asumimos que la columna del índice IPSA se llama 'Price' al cargar
    engineer = TechnicalFeatureEngineer(target_price_col='Price')
    df = engineer.add_indicators(df)

    print("\n--- PASO 3: Modelado de Volatilidad (EGARCH) ---")
    # Le indicamos que calcule la volatilidad basándose en la columna 'Price'
    vol_modeler = VolatilityModeler(window_size=500, target_col='Price')
    df = vol_modeler.compute_egarch(df)

    print("\n--- PASO 4: Estacionariedad y Memoria (FFD) ---")
    differencer = FractionalDifferencer(threshold=1e-4)
    
    # Protegemos las columnas que NO deben sufrir diferenciación fraccionaria
    # (Fechas, indicadores que ya son estacionarios, y volatilidad calculada)
    columnas_intocables = [
        'Date', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'ATR', 
        'EGARCH_Vol', 'TPM', 'VIX', 'Yield10Y', 'Spread_10Y_3M'
    ]
    df_final = differencer.apply_ffd(df, columns_to_ignore=columnas_intocables)

    print("\n--- PASO 5: Guardando Resultados ---")
    # Creamos la carpeta processed si por algún motivo no existe
    os.makedirs("./data/processed", exist_ok=True)
    
    output_path = "./data/processed/ipsa_master_processed.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ ¡Pipeline completado con éxito!")
    print(f"📁 Datos limpios y estacionarios guardados en: {output_path}")

if __name__ == "__main__":
    run_preprocessing_pipeline()
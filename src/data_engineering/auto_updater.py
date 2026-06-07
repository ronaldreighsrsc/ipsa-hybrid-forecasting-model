import os
import pandas as pd
import yfinance as yf
import requests
import warnings
import urllib3
from datetime import datetime

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuración de rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Diccionario de Tickers de Yahoo Finance
YF_TICKERS = {
    # 'IPSA.csv': '^IPSA', # yfinance API no provee historial correcto, solo el último día
    'S&P 500 Historical Data.csv': '^GSPC',
    'FXI ETF Stock Price History.csv': 'FXI',
    'USD_CLP Historical Data.csv': 'CLP=X',
    'CBOE Volatility Index Historical Data.csv': '^VIX',
    'Copper.csv': 'HG=F',
    '10 Year Treasury Yield Historical Data.csv': '^TNX',
    'IRX_2008_2025.csv': '^IRX'
}

def update_yfinance_data():
    print("Iniciando actualización desde Yahoo Finance...")
    for filename, ticker in YF_TICKERS.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[WARNING] Archivo {filename} no encontrado, omitiendo.")
            continue
            
        print(f"Descargando datos recientes para {ticker}...")
        
        try:
            # Cargar archivo existente para saber la última fecha
            df_existing = pd.read_csv(filepath)
            
            # Detectar formato de fecha del archivo existente
            if '-' in str(df_existing['Date'].iloc[0]):
                date_format = '%Y-%m-%d'
            else:
                date_format = '%m/%d/%Y'
                
            df_existing['Date'] = pd.to_datetime(df_existing['Date'], errors='coerce')
            last_date = df_existing['Date'].max()
            
            # Descargar datos faltantes
            start_date_str = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            data = yf.download(ticker, start=start_date_str, progress=False)
            
            if data.empty:
                print(f"[WARNING] No se encontraron datos recientes para {ticker}.")
                continue
                
            # Aplanar el multi-index de yfinance si existe
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Formatear el DataFrame descargado
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None) # Quitar zona horaria
            
            # Ya se cargó el archivo existente arriba
            
            # Filtrar datos nuevos
            new_data = data[data['Date'] > last_date]
            
            if new_data.empty:
                print(f"[OK] {filename} ya est\u00E1 actualizado (\u00DAltima fecha: {last_date.date()}).")
                continue
                
            # Adaptar columnas al formato del CSV
            if filename == 'IPSA.csv':
                new_rows = new_data[['Date', 'Close', 'High', 'Low', 'Open']].copy()
                new_rows.rename(columns={'Close': 'Price'}, inplace=True)
            else:
                new_rows = new_data[['Date', 'Close']].copy()
                new_rows.rename(columns={'Close': 'Price'}, inplace=True)
                
            # Convertir fechas de vuelta a string con el formato original
            df_existing['Date'] = df_existing['Date'].dt.strftime(date_format)
            new_rows['Date'] = new_rows['Date'].dt.strftime(date_format)
            
            # Formatear números con separador de miles si el original lo tiene
            if df_existing['Price'].dtype == 'object' and ',' in str(df_existing['Price'].iloc[0]):
                for col in new_rows.columns:
                    if col != 'Date':
                        new_rows[col] = new_rows[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
            else:
                 # Si el original usa punto decimal, asegurar redondeo a 2 o 4 decimales
                 for col in new_rows.columns:
                    if col != 'Date':
                        new_rows[col] = new_rows[col].round(4)
            
            # Concatenar y guardar
            df_updated = pd.concat([df_existing, new_rows], ignore_index=True)
            df_updated.to_csv(filepath, index=False)
            print(f"[EXITO] {filename} actualizado con {len(new_rows)} nuevas filas.")
            
        except Exception as e:
            print(f"[ERROR] Error actualizando {ticker}: {e}")

def update_embi():
    print("\nIniciando actualización de EMBI...")
    url = "https://bcrdgdcprod.blob.core.windows.net/documents/entorno-internacional/documents/Serie_Historica_Spread_del_EMBI.xlsx"
    filepath = os.path.join(RAW_DATA_DIR, 'Serie_Historica_Spread_del_EMBI.xlsx')
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print("[EXITO] EMBI actualizado exitosamente mediante descarga directa.")
    except Exception as e:
        print(f"[ERROR] Error descargando EMBI: {e}")

def update_tpm():
    print("\nIniciando actualización de TPM...")
    filepath = os.path.join(RAW_DATA_DIR, 'TPM.xlsx')
    url = "https://mindicador.cl/api/tpm"
    
    try:
        # 1. Obtener TPM de la API mindicador.cl
        res = requests.get(url, verify=False).json()
        latest_data = res['serie'][0]
        latest_date = pd.to_datetime(latest_data['fecha']).tz_localize(None)
        latest_tpm = latest_data['valor']
        
        # 2. Leer Excel local
        if not os.path.exists(filepath):
            print("[WARNING] Archivo TPM.xlsx no encontrado.")
            return
            
        df_tpm = pd.read_excel(filepath)
        df_tpm['Date'] = pd.to_datetime(df_tpm['Date'])
        local_last_date = df_tpm['Date'].max()
        
        # 3. Comparar fechas (solo comparar fecha, ignorar hora)
        if latest_date.date() > local_last_date.date():
            print(f"Actualizando TPM: Nuevo dato encontrado ({latest_tpm}%) para la fecha {latest_date.date()}")
            new_row = pd.DataFrame({'Date': [latest_date], 'TPM': [latest_tpm]})
            df_updated = pd.concat([df_tpm, new_row], ignore_index=True)
            df_updated.to_excel(filepath, index=False)
            print("[EXITO] TPM actualizado exitosamente.")
        else:
            print(f"[OK] TPM ya est\u00E1 actualizado (\u00DAltimo dato local: {local_last_date.date()}, API: {latest_date.date()}).")
            
    except Exception as e:
        print(f"[ERROR] Error actualizando TPM: {e}")

if __name__ == "__main__":
    print(f"=== INICIANDO PIPELINE DE DATOS ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    update_yfinance_data()
    update_embi()
    update_tpm()
    print("=== PIPELINE FINALIZADO ===")

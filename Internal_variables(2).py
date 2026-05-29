import pandas as pd
import numpy as np
import os

# --- SOLUCIÓN DE RUTAS
# Obtiene la ruta de la carpeta exacta donde está guardado este script
script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_master = os.path.join(script_dir, 'Base_Datos_Tesis_Master.csv')
ruta_ipsa = os.path.join(script_dir, 'IPSA.csv')

# ==========================================
# 1. CARGAR DATOS
# ==========================================
df_master = pd.read_csv(ruta_master)
df_ipsa = pd.read_csv(ruta_ipsa, thousands=',')

df_master['Date'] = pd.to_datetime(df_master['Date'])
df_ipsa['Date'] = pd.to_datetime(df_ipsa['Date'])
df_ipsa = df_ipsa.sort_values('Date').reset_index(drop=True)

# ==========================================
# 2. CÁLCULO DE INDICADORES (SIN MULTICOLINEALIDAD)
# ==========================================

# --- 1. MOMENTUM: MACD (12, 26, 9) ---
ema12 = df_ipsa['Price'].ewm(span=12, adjust=False).mean()
ema26 = df_ipsa['Price'].ewm(span=26, adjust=False).mean()
df_ipsa['MACD'] = ema12 - ema26
df_ipsa['MACD_Signal'] = df_ipsa['MACD'].ewm(span=9, adjust=False).mean()
df_ipsa['MACD_Hist'] = df_ipsa['MACD'] - df_ipsa['MACD_Signal']

# --- 2. MOMENTUM: RSI (14 periodos - FÓRMULA DE WILDER) ---
delta = df_ipsa['Price'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
# Wilder usa EWM con alpha = 1/window para suavizado exponencial
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
df_ipsa['RSI'] = 100 - (100 / (1 + rs))

# --- 3. VOLATILIDAD: ATR (14 periodos) ---
# El ATR usa el High, Low y el cierre previo para ver el rango real de movimiento
high_low = df_ipsa['High'] - df_ipsa['Low']
high_close = np.abs(df_ipsa['High'] - df_ipsa['Price'].shift())
low_close = np.abs(df_ipsa['Low'] - df_ipsa['Price'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = ranges.max(axis=1)
# Suavizado de Wilder
df_ipsa['ATR'] = true_range.ewm(alpha=1/14, adjust=False).mean()

# ==========================================
# 3. UNIR A LA BASE MAESTRA
# ==========================================
# Agregamos solo las variables definitivas (Sin CCI ni BBW)
cols_to_add = ['Date', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'ATR']
df_indicators = df_ipsa[cols_to_add]

df_final = pd.merge(df_master, df_indicators, on='Date', how='left')

# ==========================================
# 4. LIMPIEZA FINAL Y GUARDAR
# ==========================================
# Eliminamos filas del periodo de calentamiento inicial
# El MACD necesita 26 días para estabilizarse, usamos MACD_Hist como referencia
df_final = df_final.dropna(subset=['MACD_Hist', 'RSI', 'ATR'])

print("Indicadores técnicos definitivos calculados (Sin multicolinealidad).")
print(df_final[['Date', 'Price', 'RSI', 'MACD_Hist', 'ATR']].head())

# Sobrescribimos el master usando la ruta segura
df_final.to_csv(ruta_master, index=False)
print(f"\n📁 Base de datos actualizada y lista para FFD en:\n{ruta_master}")
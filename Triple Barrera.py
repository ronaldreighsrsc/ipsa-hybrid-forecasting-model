import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, skew, kurtosis
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN DEL TORNEO ---
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_RAW = os.path.join(script_dir, 'Base_Datos_Tesis_Master.csv')

# --- PARÁMETROS DE FRICCIÓN (REALISMO INSTITUCIONAL) ---
COMISION_BRÓKER = 0.0005  #0.05% (Nivel institucional o Trii Pro).
SLIPPAGE_ESTIMADO = 0.0003
COSTO_POR_MOVIMIENTO = COMISION_BRÓKER + SLIPPAGE_ESTIMADO # Se aplica al entrar Y al salir

K_UP = 2.0     
K_DOWN = 1.5   
MAX_HOLD = 10 

#MODELOS = ['arimax', 'lstm', 'bilstm', 'rf', 'xgb', 'lstm_rf', 'arima_lstm']
MODELOS = ['arimax','lstm','bilstm','rf','xgb','lstm_rf','arima_lstm']
BANCOS = ['Univariado', 'Precio_Puro', 'Macros', 'Global', 'Tecnicos', 'Hibrido_Precio_Tec', 'Kitchen_Sink_Total']

# ==============================================================================
# 2. FUNCIONES BASE (MÉTRICAS Y TBM CON FRICCIÓN)
# ==============================================================================
def calculate_advanced_metrics(returns, mkt_dates, cum_returns, confidence=0.95):
    if len(returns) == 0: return {k: 0 for k in ['Sharpe', 'VaR_95', 'CVaR_95', 'STARR', 'PSR', 'MDD']}
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    fecha_inicio, fecha_fin = mkt_dates[0], mkt_dates[-1]
    anios_backtest = (fecha_fin - fecha_inicio).days / 365.25 
    trades_por_anio = len(returns) / anios_backtest if anios_backtest > 0 else 0
    
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_por_anio) if std_ret != 0 else 0
    
    var_level = np.percentile(returns, (1 - confidence) * 100)
    cvar_level = returns[returns <= var_level].mean() if len(returns[returns <= var_level]) > 0 else 0
    starr = mean_ret / abs(cvar_level) if cvar_level != 0 else 0
    
    n = len(returns)
    skew_ret, kurt_ret = skew(returns), kurtosis(returns)
    sigma_sr = np.sqrt((1 / (n - 1)) * (1 + 0.5 * sharpe**2 - skew_ret * sharpe + (kurt_ret / 4) * sharpe**2))
    psr = norm.cdf(sharpe / sigma_sr) if sigma_sr != 0 else 0
    
    cum_series = pd.Series(cum_returns)
    running_max = cum_series.cummax()
    drawdown = (cum_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {'Sharpe': sharpe, 'VaR_95': var_level, 'CVaR_95': cvar_level, 'STARR': starr, 'PSR': psr, 'MDD': max_drawdown}

def run_triple_barrier_simulation_with_friction(df, probabilities):
    # EXTRAEMOS HIGH Y LOW (Asegúrate de que tu CSV tenga estas columnas)
    opens = df['Open'].values
    closes = df['Price'].values
    highs = df['High'].values  # <- Precio Máximo del día
    lows = df['Low'].values    # <- Precio Mínimo del día
    vols = df['EGARCH_Vol'].values
    dates = df.index
    results = []
    
    for i in range(len(closes) - MAX_HOLD - 1):
        if probabilities[i] > 0.5:
            entry_price = opens[i + 1] 
            vol_entry = vols[i] 
            
            tp_level = entry_price * (1 + K_UP * (vol_entry/100))
            sl_level = entry_price * (1 - K_DOWN * (vol_entry/100))
            
            for j in range(1, MAX_HOLD + 1):
                curr_close = closes[i + j]
                curr_high = highs[i + j]
                curr_low = lows[i + j]
                
                # --- VERIFICACIÓN INTRADÍA ---
                hit_tp = curr_high >= tp_level
                hit_sl = curr_low <= sl_level
                
                # Caso 1: Choque Doble (Asumimos el peor escenario: tocó el SL primero)
                if hit_tp and hit_sl:
                    raw_ret = (sl_level / entry_price) - 1
                    net_ret = (1 + raw_ret) * (1 - COSTO_POR_MOVIMIENTO) * (1 - COSTO_POR_MOVIMIENTO) - 1
                    results.append({'date': dates[i+j], 'ret': net_ret, 'type': 'SL (Dual)'})
                    break
                
                # Caso 2: Toca solo el Take Profit
                elif hit_tp:
                    raw_ret = (tp_level / entry_price) - 1
                    net_ret = (1 + raw_ret) * (1 - COSTO_POR_MOVIMIENTO) * (1 - COSTO_POR_MOVIMIENTO) - 1
                    results.append({'date': dates[i+j], 'ret': net_ret, 'type': 'TP'})
                    break
                    
                # Caso 3: Toca solo el Stop Loss
                elif hit_sl:
                    raw_ret = (sl_level / entry_price) - 1
                    net_ret = (1 + raw_ret) * (1 - COSTO_POR_MOVIMIENTO) * (1 - COSTO_POR_MOVIMIENTO) - 1
                    results.append({'date': dates[i+j], 'ret': net_ret, 'type': 'SL'})
                    break
                    
                # Caso 4: Se acaba el tiempo (MAX_HOLD), salimos al precio de cierre
                elif j == MAX_HOLD:
                    raw_ret = (curr_close / entry_price) - 1
                    net_ret = (1 + raw_ret) * (1 - COSTO_POR_MOVIMIENTO) * (1 - COSTO_POR_MOVIMIENTO) - 1
                    results.append({'date': dates[i+j], 'ret': net_ret, 'type': 'TIME'})
                    break
                    
    return pd.DataFrame(results)

# ==============================================================================
# 3. EJECUCIÓN DEL TORNEO (OPTIMIZACIÓN POR ALPHA NETO)
# ==============================================================================
print(f"🏆 INICIANDO TORNEO NETO (Fricción: {COSTO_POR_MOVIMIENTO*2:.2%} por Trade completo) 🏆\n")
df_raw = pd.read_csv(FILE_RAW, parse_dates=['Date']).sort_values('Date').set_index('Date')

campeones = {}

for modelo in MODELOS:
    mejor_alpha = -np.inf
    campeon_actual = None
    
    for banco in BANCOS:
        file_name = os.path.join(script_dir, f'pred_probs_{modelo}_{banco}.npy')
        
        if os.path.exists(file_name):
            pred_probs = np.load(file_name)
            n_test = len(pred_probs)
            df_backtest = df_raw.iloc[-n_test:].copy()
            
            # Ejecutar con fricción
            trade_results = run_triple_barrier_simulation_with_friction(df_backtest, pred_probs)
            
            if not trade_results.empty:
                trade_results = trade_results.sort_values('date').reset_index(drop=True)
                trade_results['cum_ret'] = (1 + trade_results['ret']).cumprod()
                retorno_estrategia = trade_results['cum_ret'].iloc[-1] - 1
                
                precio_inicio_mkt = df_backtest['Price'].iloc[0]
                precio_final_mkt = df_backtest['Price'].iloc[-1]
                retorno_mercado = (precio_final_mkt / precio_inicio_mkt) - 1
                
                alpha = retorno_estrategia - retorno_mercado
                
                if alpha > mejor_alpha:
                    mejor_alpha = alpha
                    metrics = calculate_advanced_metrics(trade_results['ret'].values, df_backtest.index, trade_results['cum_ret'].values)
                    mkt_equity = df_backtest['Price'] / precio_inicio_mkt
                    mkt_drawdown = (mkt_equity - mkt_equity.cummax()) / mkt_equity.cummax()
                    
                    campeon_actual = {
                        'banco': banco,
                        'alpha': alpha,
                        'ret_est': retorno_estrategia,
                        'trades': len(trade_results),
                        'win_rate': len(trade_results[trade_results['ret'] > 0]) / len(trade_results),
                        'metrics': metrics,
                        'df_trades': trade_results[['date', 'cum_ret']].copy(),
                        'df_mkt': mkt_equity.copy()
                    }
                    
    if campeon_actual:
        campeones[modelo] = campeon_actual
        print(f"🥇 Campeón {modelo.upper()}: {campeon_actual['banco']} | Alpha Neto: {campeon_actual['alpha']:.2%}")

# ==============================================================================
# 4. TABLA COMPARATIVA FINAL (INCLUYE BENCHMARK)
# ==============================================================================
print("\n" + "="*105)
print(f"{'MODELO':<12} | {'BANCO GANADOR':<18} | {'ALPHA NETO':<10} | {'RET_NETO':<9} | {'TRADES':<6} | {'WIN_%':<6} | {'SHARPE':<7} | {'MDD'} | {'CVaR_95'} | {'STARR'}")
print("-" * 105)

# 4.1 Calcular métricas para el Benchmark (IPSA puro)
if campeones:
    # Usamos el histórico del primer modelo para que las fechas coincidan
    primer_mod = list(campeones.keys())[0]
    mkt_equity_vals = campeones[primer_mod]['df_mkt'].values
    # Los retornos diarios del mercado para métricas
    mkt_returns = np.diff(mkt_equity_vals) / mkt_equity_vals[:-1]
    mkt_metrics = calculate_advanced_metrics(mkt_returns, df_backtest.index, mkt_equity_vals)
    
    # Imprimir Fila del Benchmark
    print(f"{'BENCHMARK':<12} | {'IPSA (B&H)':<18} | {'0.00%':>10} | {mkt_equity_vals[-1]-1:>8.2%} | {'-':>6} | {'-':>6} | {mkt_metrics['Sharpe']:>7.2f} | {mkt_metrics['MDD']:>7.2%} | {mkt_metrics['CVaR_95']:>7.2%} | {mkt_metrics['STARR']:>7.2f}")
    print("-" * 105)

# 4.2 Imprimir Modelos
for mod, data in campeones.items():
    m = data['metrics']
    print(f"{mod.upper():<12} | {data['banco']:<18} | {data['alpha']:>9.2%} | {data['ret_est']:>8.2%} | {data['trades']:>6} | {data['win_rate']:>5.1%} | {m['Sharpe']:>7.2f} | {m['MDD']:>7.2%} | {m['CVaR_95']:>7.2%} | {m['STARR']:>7.2f}")
print("="*105)

# ==============================================================================
# 5. GRÁFICO MAESTRO MEJORADO
# ==============================================================================
plt.figure(figsize=(14, 8))
if campeones:
    mkt_base = campeones[primer_mod]['df_mkt']
    plt.plot(mkt_base.index, mkt_base.values, color='black', label='Benchmark IPSA (Buy & Hold)', linestyle='--', linewidth=2.5, zorder=5)

# Colores y estilo
for idx, (mod, data) in enumerate(campeones.items()):
    df_t = data['df_trades']
    # Sincronizamos el inicio para que todos partan de 1.0 en la primera fecha de test
    plt.plot(df_t['date'], df_t['cum_ret'], label=f"{mod.upper()} | Alpha: {data['alpha']:.1%}", linewidth=1.5, alpha=0.9)

plt.title(f'Evaluación de Desempeño Neto (Fricción: {COSTO_POR_MOVIMIENTO*2:.2%})', fontsize=14, pad=20)
plt.ylabel('Valor de la Inversión (Base 1.0)', fontsize=12)
plt.xlabel('Periodo de Evaluación', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Estrategias", frameon=True)
plt.grid(True, which='major', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, skew, kurtosis
import warnings

warnings.filterwarnings("ignore")

class TripleBarrierBacktester:
    """
    Motor de Backtesting Financiero.
    Simula inversiones reales usando las probabilidades generadas por los modelos,
    aplicando comisiones, slippage y límites de volatilidad.
    """
    def __init__(self, data_path: str, results_dir: str):
        self.data_path = data_path
        self.results_dir = results_dir
        
        # Fricción Institucional
        self.comision = 0.0005
        self.slippage = 0.0003
        self.costo_movimiento = self.comision + self.slippage
        
        # Parámetros Triple Barrera
        self.k_up = 2.0     
        self.k_down = 1.5   
        self.max_hold = 10 

    def calculate_advanced_metrics(self, returns, mkt_dates, cum_returns, confidence=0.95):
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

    def simulate_trades(self, df, probabilities):
        """Simula las operaciones día a día evaluando las 3 barreras."""
        col_price = 'Price' if 'Price' in df.columns else 'Price_FFD'
        col_open = 'Open' if 'Open' in df.columns else 'Open_FFD'
        col_high = 'High' if 'High' in df.columns else 'High_FFD'
        col_low = 'Low' if 'Low' in df.columns else 'Low_FFD'

        opens = df[col_open].values
        closes = df[col_price].values
        highs = df[col_high].values
        lows = df[col_low].values
        vols = df['EGARCH_Vol'].values
        dates = df.index
        results = []
        
        for i in range(len(closes) - self.max_hold - 1):
            if probabilities[i] > 0.5:
                entry_price = opens[i + 1] 
                vol_entry = vols[i] 
                
                tp_level = entry_price * (1 + self.k_up * (vol_entry/100))
                sl_level = entry_price * (1 - self.k_down * (vol_entry/100))
                
                for j in range(1, self.max_hold + 1):
                    curr_close = closes[i + j]
                    curr_high = highs[i + j]
                    curr_low = lows[i + j]
                    
                    hit_tp = curr_high >= tp_level
                    hit_sl = curr_low <= sl_level
                    
                    if hit_tp and hit_sl:
                        raw_ret = (sl_level / entry_price) - 1
                        net_ret = (1 + raw_ret) * (1 - self.costo_movimiento)**2 - 1
                        results.append({'date': dates[i+j], 'ret': net_ret, 'type': 'SL (Dual)'})
                        break
                    elif hit_tp:
                        raw_ret = (tp_level / entry_price) - 1
                        net_ret = (1 + raw_ret) * (1 - self.costo_movimiento)**2 - 1
                        results.append({'date': dates[i+j], 'ret': net_ret, 'type': 'TP'})
                        break
                    elif hit_sl:
                        raw_ret = (sl_level / entry_price) - 1
                        net_ret = (1 + raw_ret) * (1 - self.costo_movimiento)**2 - 1
                        results.append({'date': dates[i+j], 'ret': net_ret, 'type': 'SL'})
                        break
                    elif j == self.max_hold:
                        raw_ret = (curr_close / entry_price) - 1
                        net_ret = (1 + raw_ret) * (1 - self.costo_movimiento)**2 - 1
                        results.append({'date': dates[i+j], 'ret': net_ret, 'type': 'TIME'})
                        break
                        
        return pd.DataFrame(results)

    def run_tournament(self, modelos, bancos):
        print(f"🏆 INICIANDO TORNEO NETO (Fricción: {self.costo_movimiento*2:.2%} por Trade) 🏆\n")
        df_raw = pd.read_csv(self.data_path, parse_dates=['Date']).sort_values('Date').set_index('Date')
        
        col_price = 'Price' if 'Price' in df_raw.columns else 'Price_FFD'
        campeones = {}

        for modelo in modelos:
            mejor_alpha = -np.inf
            campeon_actual = None
            
            for banco in bancos:
                file_name = os.path.join(self.results_dir, f'probs_{modelo.lower()}_{banco}.npy')
                
                if os.path.exists(file_name):
                    pred_probs = np.load(file_name)
                    n_test = len(pred_probs)
                    df_backtest = df_raw.iloc[-n_test:].copy()
                    
                    trade_results = self.simulate_trades(df_backtest, pred_probs)
                    
                    if not trade_results.empty:
                        trade_results = trade_results.sort_values('date').reset_index(drop=True)
                        trade_results['cum_ret'] = (1 + trade_results['ret']).cumprod()
                        retorno_estrategia = trade_results['cum_ret'].iloc[-1] - 1
                        
                        precio_inicio_mkt = df_backtest[col_price].iloc[0]
                        precio_final_mkt = df_backtest[col_price].iloc[-1]
                        retorno_mercado = (precio_final_mkt / precio_inicio_mkt) - 1
                        
                        alpha = retorno_estrategia - retorno_mercado
                        
                        if alpha > mejor_alpha:
                            mejor_alpha = alpha
                            metrics = self.calculate_advanced_metrics(trade_results['ret'].values, df_backtest.index, trade_results['cum_ret'].values)
                            mkt_equity = df_backtest[col_price] / precio_inicio_mkt
                            
                            campeon_actual = {
                                'banco': banco, 'alpha': alpha, 'ret_est': retorno_estrategia,
                                'trades': len(trade_results),
                                'win_rate': len(trade_results[trade_results['ret'] > 0]) / len(trade_results),
                                'metrics': metrics,
                                'df_trades': trade_results[['date', 'cum_ret']].copy(),
                                'df_mkt': mkt_equity.copy()
                            }
                            
            if campeon_actual:
                campeones[modelo] = campeon_actual
                print(f"🥇 Campeón {modelo.upper()}: {campeon_actual['banco']} | Alpha Neto: {campeon_actual['alpha']:.2%}")

        return campeones, df_raw

    def print_summary_table(self, campeones, df_raw):
        if not campeones:
            print("⚠️ No hay campeones para mostrar en la tabla.")
            return

        print("\n" + "="*105)
        print(f"{'MODELO':<12} | {'BANCO GANADOR':<18} | {'ALPHA NETO':<10} | {'RET_NETO':<9} | {'TRADES':<6} | {'WIN_%':<6} | {'SHARPE':<7} | {'MDD'} | {'CVaR_95'} | {'STARR'}")
        print("-" * 105)

        primer_mod = list(campeones.keys())[0]
        mkt_base = campeones[primer_mod]['df_mkt']
        mkt_returns = np.diff(mkt_base.values) / mkt_base.values[:-1]
        
        mkt_metrics = self.calculate_advanced_metrics(mkt_returns, df_raw.iloc[-len(mkt_base):].index, mkt_base.values)
        
        print(f"{'BENCHMARK':<12} | {'IPSA (B&H)':<18} | {'0.00%':>10} | {mkt_base.values[-1]-1:>8.2%} | {'-':>6} | {'-':>6} | {mkt_metrics['Sharpe']:>7.2f} | {mkt_metrics['MDD']:>7.2%} | {mkt_metrics['CVaR_95']:>7.2%} | {mkt_metrics['STARR']:>7.2f}")
        print("-" * 105)

        for mod, data in campeones.items():
            m = data['metrics']
            print(f"{mod.upper():<12} | {data['banco']:<18} | {data['alpha']:>9.2%} | {data['ret_est']:>8.2%} | {data['trades']:>6} | {data['win_rate']:>5.1%} | {m['Sharpe']:>7.2f} | {m['MDD']:>7.2%} | {m['CVaR_95']:>7.2%} | {m['STARR']:>7.2f}")
        print("="*105)

    def plot_results(self, campeones):
        if not campeones:
            print("⚠️ No hay campeones para graficar.")
            return

        plt.figure(figsize=(14, 8))
        primer_mod = list(campeones.keys())[0]
        mkt_base = campeones[primer_mod]['df_mkt']
        plt.plot(mkt_base.index, mkt_base.values, color='black', label='Benchmark IPSA (Buy & Hold)', linestyle='--', linewidth=2.5, zorder=5)

        for idx, (mod, data) in enumerate(campeones.items()):
            df_t = data['df_trades']
            plt.plot(df_t['date'], df_t['cum_ret'], label=f"{mod.upper()} | Alpha: {data['alpha']:.1%}", linewidth=1.5, alpha=0.9)

        plt.title(f'Evaluación de Desempeño Neto (Fricción: {self.costo_movimiento*2:.2%})', fontsize=14, pad=20)
        plt.ylabel('Valor de la Inversión (Base 1.0)', fontsize=12)
        plt.xlabel('Periodo de Evaluación', fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Estrategias", frameon=True)
        plt.grid(True, which='major', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
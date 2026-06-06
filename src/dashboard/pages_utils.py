import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

@st.cache_data
def load_data():
    """Carga el dataset procesado."""
    filepath = 'data/processed/ipsa_master_processed.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

@st.cache_data
def load_ablation_results():
    """Carga los resultados de la tabla de ablación."""
    filepath = 'src/evaluation/results/tabla_ablacion_global.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df
    return None

def plot_line_chart(df, x_col, y_cols, title, y_label):
    """Genera un gráfico de línea interactivo con Plotly."""
    fig = px.line(df, x=x_col, y=y_cols, title=title)
    fig.update_layout(yaxis_title=y_label, hovermode='x unified')
    return fig

def plot_candlestick(df, title="Gráfico de Velas"):
    """Genera un gráfico de velas (Candlestick) con Plotly."""
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Price'])])
    fig.update_layout(
        title=title, 
        yaxis_title='Precio', 
        xaxis_rangeslider_visible=False
    )
    
    return fig

@st.cache_data
def get_backtesting_results():
    """Ejecuta el backtest en memoria y devuelve los resultados."""
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.evaluation.triple_barrera import TripleBarrierBacktester
    
    data_path = "./data/processed/ipsa_master_processed.csv"
    results_dir = "./src/evaluation/results/"
    
    if not os.path.exists(data_path) or not os.path.exists(results_dir):
        return None, None, None
        
    tester = TripleBarrierBacktester(data_path=data_path, results_dir=results_dir)
    modelos = ['arimax', 'random_forest', 'xgboost', 'lstm', 'bilstm', 'arima_lstm', 'lstm_rf']
    bancos = ['Univariado', 'Precio_Puro', 'Macros', 'Global', 'Tecnicos', 'Hibrido_Precio_Tec', 'Kitchen_Sink_Total']
    
    campeones, df_raw = tester.run_tournament(modelos, bancos)
    
    if campeones:
        primer_mod = list(campeones.keys())[0]
        mkt_base = campeones[primer_mod]['df_mkt']
        
        # Calcular metricas del benchmark
        mkt_returns = np.diff(mkt_base.values) / mkt_base.values[:-1]
        mkt_metrics = tester.calculate_advanced_metrics(mkt_returns, df_raw.iloc[-len(mkt_base):].index, mkt_base.values)
        
        # Armar la tabla resumen en formato pandas para el dashboard
        table_data = []
        table_data.append({
            'Modelo': 'IPSA (Benchmark)',
            'Banco Ganador': 'Buy & Hold',
            'Alpha Neto': '0.00%',
            'Retorno Neto': f"{(mkt_base.values[-1]-1):.2%}",
            'Trades': '-',
            'Win Rate': '-',
            'Sharpe': f"{mkt_metrics['Sharpe']:.2f}",
            'Max Drawdown': f"{mkt_metrics['MDD']:.2%}",
            'CVaR_95': f"{mkt_metrics['CVaR_95']:.2%}",
            'STARR': f"{mkt_metrics['STARR']:.2f}"
        })
        
        for mod, data in campeones.items():
            m = data['metrics']
            table_data.append({
                'Modelo': mod.upper(),
                'Banco Ganador': data['banco'],
                'Alpha Neto': f"{data['alpha']:.2%}",
                'Retorno Neto': f"{data['ret_est']:.2%}",
                'Trades': str(data['trades']),
                'Win Rate': f"{data['win_rate']:.1%}",
                'Sharpe': f"{m['Sharpe']:.2f}",
                'Max Drawdown': f"{m['MDD']:.2%}",
                'CVaR_95': f"{m['CVaR_95']:.2%}",
                'STARR': f"{m['STARR']:.2f}"
            })
            
        df_summary = pd.DataFrame(table_data)
        return df_summary, campeones, mkt_base
        
    return None, None, None

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

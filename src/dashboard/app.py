import streamlit as st
from pages_utils import load_data, load_ablation_results, plot_line_chart, plot_candlestick
import plotly.express as px
import os
from PIL import Image

# Configuración de la página ("Vitrina" mode)
st.set_page_config(page_title="IPSA Hybrid Forecasting", page_icon="📈", layout="wide")

# CSS personalizado para darle un toque premium
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        color: #1E88E5;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Navegación en la barra lateral
st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=80)
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una sección:", 
                        ["Resumen del Proyecto", "Exploración de Datos (EDA)", "Estudio de Ablación", "Backtesting Financiero"])

st.sidebar.markdown("---")
st.sidebar.info("Dashboard desarrollado como vitrina para el Modelo Híbrido de Predicción del Índice IPSA.")

# --- Página 1: Resumen del Proyecto ---
if page == "Resumen del Proyecto":
    st.markdown('<p class="main-title">Predicción Direccional del IPSA</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Modelos Híbridos Deep Learning vs Econometría Tradicional</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Acerca del Proyecto
        Este dashboard interactivo presenta los resultados de mi trabajo de final de carrera enfocado en la predicción direccional del índice bursátil chileno (IPSA).
        
        **El problema:** Predecir si el mercado subirá o bajará mañana es extremadamente difícil debido a la alta proporción de ruido vs señal.
        
        **La solución:** Un enfoque estructurado que combina:
        - **Procesamiento de vanguardia:** Filtrado de volatilidad (EGARCH) y Diferenciación Fraccionaria (FFD) para lograr estacionariedad preservando la memoria a largo plazo.
        - **Modelos de Machine Learning & Deep Learning:** Random Forest, XGBoost, LSTM y arquitecturas Híbridas (Extractor y Residual).
        - **Backtesting Realista:** Validación con el método de Triple Barrera, considerando fricción institucional y límites temporales.
        """)
    
    with col2:
        st.success("### Tecnologías Core")
        st.write("🐍 Python (Pandas, Numpy)")
        st.write("🧠 TensorFlow / Keras")
        st.write("🌲 Scikit-Learn / XGBoost")
        st.write("📉 Statsmodels / ARCH")
        st.write("📊 Streamlit / Plotly")

# --- Página 2: Exploración de Datos (EDA) ---
elif page == "Exploración de Datos (EDA)":
    st.markdown('<p class="main-title">Exploración del Test Dataset</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        st.dataframe(df.tail(100), width='stretch')
        
        # --- Filtros Interactivos ---
        st.subheader("Evolución Histórica del IPSA")
        
        col_filt1, col_filt2 = st.columns([1, 2])
        with col_filt1:
            temporalidad = st.radio("Temporalidad de Velas:", ["Diario", "Semanal", "Mensual"], horizontal=True)
        
        with col_filt2:
            min_date, max_date = df['Date'].min(), df['Date'].max()
            rango_fechas = st.slider("Rango de Fechas (Auto-escala Eje Y):", 
                                     min_value=min_date.date(), 
                                     max_value=max_date.date(), 
                                     value=(min_date.date(), max_date.date()))
            
        # Filtrar por fechas
        df_filtered = df[(df['Date'].dt.date >= rango_fechas[0]) & (df['Date'].dt.date <= rango_fechas[1])].copy()
        
        # Resamplear si es necesario (OHLC)
        if temporalidad != "Diario":
            # Set Date as index for resampling
            df_resampled = df_filtered.set_index('Date')
            freq = 'W' if temporalidad == "Semanal" else 'ME'
            
            # OHLC Aggregation logic
            df_filtered = df_resampled.resample(freq).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Price': 'last'  # Close
            }).dropna().reset_index()
            
        fig_candle = plot_candlestick(df_filtered)
        st.plotly_chart(fig_candle, width='stretch')
        
        # --- Matriz de Correlación de Pearson ---
        st.subheader("Diagnóstico de Multicolinealidad (Matriz de Pearson)")
        
        # Replicar la lógica exacta del script original de la tesis:
        # Solo eliminar Date y MACD_Signal, conservar todas las demás (incluidas FFD)
        drop_cols = ['Date', 'MACD_Signal']
        df_corr = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        df_corr = df_corr.select_dtypes(include=['number'])
        
        # Eliminar precios crudos (redundantes con sus versiones FFD)
        raw_price_cols = ['Price', 'High', 'Low', 'Open']
        df_corr = df_corr.drop(columns=[c for c in raw_price_cols if c in df_corr.columns], errors='ignore')
        
        corr_matrix = df_corr.corr()
        
        import plotly.graph_objects as go
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',   # coolwarm equivalente en Plotly
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 9},
            hovertemplate='%{x} vs %{y}<br>Correlación: %{z:.4f}<extra></extra>'
        ))
        fig_corr.update_layout(
            height=750,
            xaxis_tickangle=-45,
            yaxis_autorange='reversed'  # Para que la diagonal vaya de arriba-izq a abajo-der
        )
        st.plotly_chart(fig_corr, width='stretch')
            
    else:
        st.error("⚠️ No se pudo cargar el archivo de datos procesados. Por favor, ejecuta primero el Bloque 1 (Preprocesamiento).")

# --- Página 3: Estudio de Ablación ---
elif page == "Estudio de Ablación":
    st.markdown('<p class="main-title">Rendimiento Predictivo</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    df_results = load_ablation_results()
    
    if df_results is not None:
        st.dataframe(df_results, width='stretch')
        
        # Gráfico de barras interactivo con las columnas reales del CSV
        fig = px.bar(df_results, x="Modelo", y="Accuracy_Test", color="Banco", barmode="group",
                     title="Comparativa de Accuracy (Test) entre Modelos y Bancos de Variables",
                     text_auto='.4f',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(
            yaxis_title="Accuracy (Test)",
            yaxis_range=[0.45, max(df_results["Accuracy_Test"]) + 0.03],
            legend_title_text="Banco de Variables",
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Tabla de mejores hiperparámetros
        st.subheader("Mejores Hiperparámetros por Configuración")
        st.dataframe(df_results[["Modelo", "Banco", "Mejores_Params", "Accuracy_Test"]].sort_values("Accuracy_Test", ascending=False), width='stretch')
        
    else:
        st.error("⚠️ No se encontró la tabla de ablación. Por favor, ejecuta el Bloque 2 (Torneo de Modelos).")

# --- Página 4: Backtesting Financiero ---
elif page == "Backtesting Financiero":
    st.markdown('<p class="main-title">Backtesting: Curvas de Equity</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evaluación por Triple Barrera</p>', unsafe_allow_html=True)
    
    st.info("""
    Las Curvas de Equity reales se generan en la consola y se plotean localmente tras ejecutar `main_evaluation.py`.
    Aquí podemos mostrar la Matriz de Correlación o agregar la funcionalidad de cargar los resultados financieros si se han exportado.
    """)
    
    corr_img_path = 'src/evaluation/results/Matriz_Correlacion_Tesis.png'
    if os.path.exists(corr_img_path):
        st.subheader("Matriz de Correlación de Probabilidades")
        image = Image.open(corr_img_path)
        st.image(image, caption='Análisis de Multicolinealidad de los Modelos', width='stretch')
    else:
        st.warning("⚠️ No se encontró la imagen de correlación. Se generará al ejecutar `main_evaluation.py`.")

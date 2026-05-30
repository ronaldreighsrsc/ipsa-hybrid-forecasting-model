import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

class CorrelationAnalyzer:
    """
    Módulo de análisis estadístico para evaluar la correlación de Pearson 
    y detectar multicolinealidad en el dataset procesado.
    """
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_and_plot(self, drop_cols: list = None, threshold: float = 0.80):
        print("📊 Iniciando Análisis de Correlación y Multicolinealidad...")
        
        # 1. Cargar datos de la arquitectura SOLID
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ No se encontró la base de datos en: {self.data_path}. Ejecuta main_preprocessing.py primero.")
        
        df = pd.read_csv(self.data_path)
        
        # 2. Limpieza de columnas no deseadas
        if drop_cols is None:
            drop_cols = ['Date', 'MACD_Signal']
        else:
            drop_cols = [col for col in drop_cols if col in df.columns]
            
        df_corr = df.drop(columns=drop_cols, errors='ignore')
        
        # Quedarse solo con variables numéricas por seguridad
        df_corr = df_corr.select_dtypes(include=[np.number])

        # 3. Matriz de Correlación de Pearson
        corr_matrix = df_corr.corr()

        # 4. Reporte de Multicolinealidad
        print(f"\n🔥 Reporte de Posible Multicolinealidad (Umbral > {threshold}):")
        
        # Extraer el triángulo superior para no repetir pares (Ej: A vs B y B vs A)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        pairs = upper_tri.stack().sort_values(ascending=False)
        
        found_collinearity = False
        for index, value in pairs.items():
            if abs(value) > threshold:
                print(f"  - {index[0]} vs {index[1]}: {value:.4f}")
                found_collinearity = True
        
        if not found_collinearity:
            print("  ✅ No se encontraron pares con alta multicolinealidad.")

        # 5. Configuración y dibujo del Mapa de Calor (Heatmap)
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, 
                    mask=None, 
                    annot=True,          
                    fmt=".2f",           
                    cmap='coolwarm',     
                    vmin=-1, vmax=1,     
                    center=0,
                    linewidths=.5,
                    cbar_kws={"shrink": .8})

        plt.title('Matriz de Correlación de Pearson: Variables y Features Técnicos', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 6. Guardar y mostrar
        output_file = os.path.join(self.output_dir, 'Matriz_Correlacion_Tesis.png')
        plt.savefig(output_file, dpi=300)
        print(f"\n✅ Matriz guardada exitosamente en: {output_file}")
        
        # Muestra el gráfico en pantalla
        plt.show()
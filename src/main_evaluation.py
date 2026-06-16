import matplotlib.pyplot as plt
import os
import json
import shutil
import warnings
from evaluation.triple_barrera import TripleBarrierBacktester
from evaluation.correlation_analyzer import CorrelationAnalyzer

warnings.filterwarnings("ignore")

def run_evaluation_pipeline():
    print("📈 Iniciando Pipeline de Evaluación Financiera (Bloque 3)...")
    
    # 1. Definir Rutas relativas a la arquitectura SOLID
    data_path = "./data/processed/ipsa_master_processed.csv"
    results_dir = "./src/evaluation/results/"
    
    # Validación rápida fail-fast
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ No se encontró la base de datos: {data_path}")
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"❌ No se encontró la carpeta de resultados: {results_dir}. Corre main_ablation.py primero.")

    # 1.5. Análisis de Correlación y Multicolinealidad
    analyzer = CorrelationAnalyzer(data_path=data_path, output_dir=results_dir)
    analyzer.analyze_and_plot()

    # 2. Inicializar el Motor de Backtesting
    tester = TripleBarrierBacktester(data_path=data_path, results_dir=results_dir)
    
    # 3. Definir Participantes del Torneo
    # NOTA: En main_ablation.py los guardamos en minúsculas (probs_arimax_..., probs_random_forest_...)
    modelos = [
        'arimax', 
        'random_forest', 
        'xgboost', 
        'lstm', 
        'bilstm', 
        'arima_lstm', 
        'lstm_rf'
    ]
    
    # Asegúrate de que esta lista contenga los bancos que corriste en main_ablation.py
    bancos = [
        'Univariado', 
        'Precio_Puro',
        'Macros', 
        'Global', 
        'Tecnicos', 
        'Hibrido_Precio_Tec',
        'Kitchen_Sink_Total'
    ]

    # 4. Ejecutar el Backtest
    campeones, df_raw = tester.run_tournament(modelos, bancos)

    # 5. Imprimir Tabla de Resumen y Exportar Model Registry
    if campeones:
        tester.print_summary_table(campeones, df_raw)
        
        print("\n💾 Guardando Model Registry...")
        os.makedirs("./models", exist_ok=True)
        registry = {}
        for mod, data in campeones.items():
            registry[f"{mod.lower()}"] = {
                "banco": data['banco'],
                "alpha": data['alpha'],
                "ret_est": data['ret_est'],
                "trades": data['trades'],
                "win_rate": data['win_rate'],
                "metrics": data['metrics'] # Incluye Sharpe, VaR, MDD, etc.
            }
            
        with open("./models/registry.json", "w") as f:
            json.dump(registry, f, indent=4)
        print("✅ Registry exportado en ./models/registry.json")

        # 6. Graficar Resultados
        print("📊 Generando gráfico maestro de rendimiento (Cierra la ventana del gráfico para terminar el script)...")
        tester.plot_results(campeones)
    else:
        print("⚠️ No se encontraron predicciones exitosas (Alpha positivo) o no hay archivos .npy para evaluar.")

if __name__ == "__main__":
    run_evaluation_pipeline()
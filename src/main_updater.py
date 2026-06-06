import sys
import os
from datetime import datetime

# Añadir el directorio 'src' al path temporalmente para asegurar que los imports funcionen
# independientemente de si el script se llama desde root o desde src/
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Importamos las funciones principales de nuestros módulos (POO)
from data_engineering.scrape_ipsa import scrape_ipsa_investing
from data_engineering.auto_updater import update_yfinance_data, update_embi, update_tpm

def run_step(description, func):
    print(f"\n{'='*50}")
    print(f"🚀 INICIANDO: {description}")
    print(f"{'='*50}")
    
    try:
        func()
        print(f"\n✅ ÉXITO: {description} completado.")
    except Exception as e:
        print(f"\n❌ ERROR: {description} falló. Detalle: {e}")
        sys.exit(1)

def main():
    # Asegurar que la consola de Windows no crashee al imprimir emojis u otros caracteres especiales
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("🌟 BIENVENIDO AL SISTEMA DE ACTUALIZACIÓN DE DATOS CRUDOS 🌟")
    print("Este script actualizará toda la base de datos cruda desde internet.\n")
    
    # 1. Scraping del IPSA (Local Selenium)
    run_step("Paso 1: Extracción de IPSA desde Investing.com (Evasión Cloudflare)", scrape_ipsa_investing)
    
    # 2. Descarga del resto de variables (API Yfinance / FRED)
    def update_apis():
        print(f"=== INICIANDO PIPELINE DE DATOS ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
        update_yfinance_data()
        update_embi()
        update_tpm()
        print("=== PIPELINE FINALIZADO ===")
        
    run_step("Paso 2: Extracción de S&P500, Cobre, TPM, FXI, etc. vía APIs", update_apis)
    
    print(f"\n{'='*50}")
    print("🎉 ACTUALIZACIÓN DE DATOS CRUDOS COMPLETADA CON ÉXITO 🎉")
    print("NOTA: Recuerda ejecutar 'python src/main_preprocessing.py' si deseas entrenar modelos.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()

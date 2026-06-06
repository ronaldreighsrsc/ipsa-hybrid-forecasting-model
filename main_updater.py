import sys
import os
import subprocess

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"🚀 INICIANDO: {description}")
    print(f"{'='*50}")
    
    try:
        # Ejecutar el script capturando la salida en tiempo real
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        return_code = process.poll()
        
        if return_code == 0:
            print(f"\n✅ ÉXITO: {description} completado.")
        else:
            print(f"\n❌ ERROR: {description} falló con código {return_code}.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ EXCEPCIÓN: Falló la ejecución de {description}. Detalle: {e}")
        sys.exit(1)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("🌟 BIENVENIDO AL SISTEMA DE ACTUALIZACIÓN MAESTRO DEL IPSA 🌟")
    print("Este script actualizará toda la base de datos y preprocesará las variables.\n")
    
    # 1. Scraping del IPSA (Local Selenium)
    run_command(
        f"{sys.executable} src/data_engineering/scrape_ipsa.py", 
        "Paso 1: Extracción de IPSA desde Investing.com (Evasión Cloudflare)"
    )
    
    # 2. Descarga del resto de variables (API Yfinance / FRED)
    run_command(
        f"{sys.executable} src/data_engineering/auto_updater.py", 
        "Paso 2: Extracción de S&P500, Cobre, TPM, FXI, etc. vía APIs"
    )
    
    # 3. Consolidación y Preprocesamiento (EGARCH, FFD, etc)
    # Importante: seteamos PYTHONIOENCODING para evitar errores de caracteres en Windows
    run_command(
        f"set PYTHONIOENCODING=utf-8 && {sys.executable} src/main_preprocessing.py", 
        "Paso 3: Preprocesamiento y Consolidación (main_preprocessing.py)"
    )
    
    print(f"\n{'='*50}")
    print("🎉 ACTUALIZACIÓN MAESTRA COMPLETADA CON ÉXITO 🎉")
    print("Todos los datos están listos para ser usados por main_ablation.py o main_evaluation.py.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()

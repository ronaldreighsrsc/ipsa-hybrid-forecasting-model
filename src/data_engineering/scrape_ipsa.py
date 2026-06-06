import os
import time
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from datetime import datetime

def scrape_ipsa_investing():
    """
    Scrapes IPSA historical data from Investing.com using undetected-chromedriver.
    """
    print("🚀 Iniciando Scraper de IPSA en Investing.com...")
    
    url = "https://www.investing.com/indices/ipsa-historical-data"
    
    options = uc.ChromeOptions()
    options.headless = False # Es mejor dejarlo visible para evitar detección de bots, aunque se puede probar True
    
    try:
        driver = uc.Chrome(options=options)
        driver.get(url)
        print("Esperando a que cargue la tabla de datos...")
        time.sleep(10) # Esperar a que pase Cloudflare y cargue la tabla
        
        html = driver.page_source
        driver.quit()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Buscar la tabla de datos históricos
        # Usualmente Investing tiene una tabla con clases específicas, buscaremos por cabeceras
        tables = soup.find_all('table')
        target_table = None
        for table in tables:
            if 'Date' in table.text and 'Price' in table.text and 'Open' in table.text:
                target_table = table
                break
                
        if not target_table:
            print("❌ No se encontró la tabla de datos. Cloudflare podría estar bloqueando.")
            return None
            
        rows = target_table.find('tbody').find_all('tr')
        
        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                # El formato suele ser: Date, Price, Open, High, Low, Vol., Change %
                date_str = cols[0].text.strip()
                price = cols[1].text.strip().replace(',', '')
                open_p = cols[2].text.strip().replace(',', '')
                high = cols[3].text.strip().replace(',', '')
                low = cols[4].text.strip().replace(',', '')
                
                try:
                    # Parsear la fecha de 'Jun 05, 2026' a '06/05/2026' (formato de IPSA.csv)
                    parsed_date = datetime.strptime(date_str, '%b %d, %Y')
                    formatted_date = parsed_date.strftime('%m/%d/%Y')
                    
                    data.append({
                        'Date': formatted_date,
                        'Price': price,
                        'Open': open_p,
                        'High': high,
                        'Low': low
                    })
                except ValueError:
                    continue
                    
        df_new = pd.DataFrame(data)
        
        if df_new.empty:
            print("❌ No se extrajeron datos.")
            return None
            
        print(f"✅ Se extrajeron {len(df_new)} filas recientes de Investing.com.")
        
        # Guardar o actualizar IPSA.csv
        file_path = 'data/raw/IPSA.csv'
        
        # Renombramos y ordenamos las columnas según el CSV original: Date, Price, High, Low, Open
        df_new = df_new[['Date', 'Price', 'High', 'Low', 'Open']]
        
        if os.path.exists(file_path):
            df_old = pd.read_csv(file_path)
            # Combinar y eliminar duplicados basados en 'Date'
            df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['Date'], keep='last')
            # Ordenar por fecha si es necesario (el original está de más antiguo a más reciente)
            df_combined['Date_dt'] = pd.to_datetime(df_combined['Date'], format='%m/%d/%Y')
            df_combined = df_combined.sort_values('Date_dt').drop(columns=['Date_dt'])
            
            df_combined.to_csv(file_path, index=False)
            print(f"✅ Archivo {file_path} actualizado exitosamente. Filas totales: {len(df_combined)}")
        else:
            # Si no existe, al menos guardamos esto, aunque en el proyecto siempre debería existir
            df_new['Date_dt'] = pd.to_datetime(df_new['Date'], format='%m/%d/%Y')
            df_new = df_new.sort_values('Date_dt').drop(columns=['Date_dt'])
            df_new.to_csv(file_path, index=False)
            print(f"✅ Archivo {file_path} creado exitosamente.")
            
        return df_new
        
    except Exception as e:
        print(f"❌ Error durante el scraping: {e}")
        try:
            driver.quit()
        except:
            pass
        return None

if __name__ == "__main__":
    scrape_ipsa_investing()

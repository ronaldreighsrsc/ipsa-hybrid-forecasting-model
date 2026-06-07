import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

class DataLoader:
    """
    Clase responsable exclusivamente de la ingesta, estandarización 
    y consolidación de los datos crudos (Raw Data).
    """
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def _read_file(self, filename: str, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """Método interno de seguridad para leer archivos."""
        path = os.path.join(self.raw_data_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: No se encontró el archivo {path}. Revisa la carpeta data/raw/")
        
        if file_type == 'csv':
            return pd.read_csv(path, **kwargs)
        return pd.read_excel(path, **kwargs)

    def load_and_merge(self) -> pd.DataFrame:
        print("Iniciando extracción y consolidación de datos crudos...")
        
        # 1. Cargas Básicas
        df_ipsa = self._read_file('IPSA.csv', thousands=',')
        df_embi = self._read_file('Serie_Historica_Spread_del_EMBI.xlsx', file_type='excel', skiprows=1)
        df_sp500 = self._read_file('S&P 500 Historical Data.csv', thousands=',')
        df_fxi = self._read_file('FXI ETF Stock Price History.csv', thousands=',')
        df_usdclp = self._read_file('USD_CLP Historical Data.csv', thousands=',')
        df_vix = self._read_file('CBOE Volatility Index Historical Data.csv', decimal='.')
        df_copper = self._read_file('Copper.csv', thousands=',')
        
        df_tpm = self._read_file('TPM.xlsx', file_type='excel')
        df_tpm = df_tpm.sort_values('Date').reset_index(drop=True)
        df_tpm['TPM'] = df_tpm['TPM'].ffill()
        
        df_yield10y = self._read_file('10 Year Treasury Yield Historical Data.csv')
        df_rate3m = self._read_file('IRX_2008_2025.csv')

        # 2. Estandarizar columnas (Renombramientos)
        df_embi = df_embi[['Fecha', 'Chile']].rename(columns={'Fecha': 'Date', 'Chile': 'EMBI'})
        df_sp500 = df_sp500.rename(columns={'Price': 'SP500'})
        df_fxi = df_fxi.rename(columns={'Price': 'FXI'})
        df_usdclp = df_usdclp.rename(columns={'Price': 'USDCLP'})
        df_vix = df_vix.rename(columns={'Price': 'VIX'})
        df_yield10y = df_yield10y.rename(columns={'Price': 'Yield10Y'})
        df_rate3m = df_rate3m.rename(columns={'Price': 'Rate_3M'})
        df_copper = df_copper.rename(columns={'Price': 'Copper'})

        # 3. Formateo de Fechas
        df_ipsa['Date'] = pd.to_datetime(df_ipsa['Date'])
        df_embi['Date'] = pd.to_datetime(df_embi['Date'], errors='coerce')
        df_sp500['Date'] = pd.to_datetime(df_sp500['Date'])
        df_fxi['Date'] = pd.to_datetime(df_fxi['Date'])
        df_usdclp['Date'] = pd.to_datetime(df_usdclp['Date'])
        df_vix['Date'] = pd.to_datetime(df_vix['Date'], format='%m/%d/%Y')
        df_yield10y['Date'] = pd.to_datetime(df_yield10y['Date'])
        df_rate3m['Date'] = pd.to_datetime(df_rate3m['Date'])
        df_tpm['Date'] = pd.to_datetime(df_tpm['Date'])
        df_copper['Date'] = pd.to_datetime(df_copper['Date'])

        # 4. Ordenar y Unir (Merge)
        dfs_to_sort = [df_ipsa, df_embi, df_sp500, df_fxi, df_usdclp, df_vix, df_yield10y, df_tpm, df_rate3m, df_copper]
        for d in dfs_to_sort:
            d.sort_values('Date', inplace=True)

        df_final = pd.merge(df_ipsa, df_embi, on='Date', how='left')
        df_final = pd.merge(df_final, df_sp500[['Date', 'SP500']], on='Date', how='left')
        df_final = pd.merge(df_final, df_fxi[['Date', 'FXI']], on='Date', how='left')
        df_final = pd.merge(df_final, df_usdclp[['Date', 'USDCLP']], on='Date', how='left')
        df_final = pd.merge(df_final, df_vix[['Date', 'VIX']], on='Date', how='left')
        df_final = pd.merge(df_final, df_yield10y[['Date', 'Yield10Y']], on='Date', how='left')
        df_final = pd.merge(df_final, df_tpm[['Date', 'TPM']], on='Date', how='left')
        df_final = pd.merge(df_final, df_rate3m[['Date', 'Rate_3M']], on='Date', how='left')
        df_final = pd.merge(df_final, df_copper[['Date', 'Copper']], on='Date', how='left')

        # 5. Cálculos y Limpieza de Feriados/Huecos
        df_final['Spread_10Y_3M'] = df_final['Yield10Y'] - df_final['Rate_3M']
        df_final['EMBI'] = pd.to_numeric(df_final['EMBI'], errors='coerce')
        
        cols_to_fill = ['EMBI', 'SP500', 'FXI', 'USDCLP', 'VIX', 'Yield10Y', 'Rate_3M', 'Spread_10Y_3M', 'TPM', 'Copper']
        df_final[cols_to_fill] = df_final[cols_to_fill].ffill()

        df_final = df_final.drop(columns=['Rate_3M'])
        df_final = df_final.loc[:, ~df_final.columns.str.contains('^Unnamed')]

        print(f"✅ Merge exitoso. Total de filas: {len(df_final)}")
        print(f"📅 Rango de fechas: {df_final['Date'].min().strftime('%Y-%m-%d')} a {df_final['Date'].max().strftime('%Y-%m-%d')}")
        print(f"📊 Total de filas: {len(df_final)}")
        return df_final

# --- TEST RÁPIDO (Solo se ejecuta si corres este archivo directamente) ---
if __name__ == "__main__":
    # La ruta asume que estás corriendo el script desde la raíz del proyecto
    loader = DataLoader(raw_data_path='./data/raw/')
    df = loader.load_and_merge()
    print(df.head())


    #venv\Scripts\activate
    #streamlit run src/dashboard/app.py
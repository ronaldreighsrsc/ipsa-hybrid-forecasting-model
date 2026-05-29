import pandas as pd
import numpy as np
import warnings
from arch import arch_model

warnings.filterwarnings("ignore")

class VolatilityModeler:
    """
    Calcula la volatilidad condicional de la serie financiera usando un modelo EGARCH
    con ventana móvil (Rolling Window). Implementa un sistema de respaldo (fallback)
    hacia GARCH simple o Volatilidad Histórica en caso de inestabilidad matemática.
    """
    def __init__(self, window_size: int = 500, target_col: str = 'Price'):
        """
        Inicializa el modelador.
        :param window_size: Días para la ventana móvil (ej. 500 días).
        :param target_col: Columna sobre la cual calcular los retornos.
        """
        self.window_size = window_size
        self.target_col = target_col

    def compute_egarch(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Calculando Volatilidad Condicional EGARCH (Ventana: {self.window_size} días)...")
        print("⏳ Esto puede tomar un par de minutos debido a las iteraciones matemáticas de la ventana móvil.")
        
        df_calc = df.copy()
        
        # 1. Calcular retornos logarítmicos (necesarios para modelos ARCH/GARCH)
        df_calc['log_ret'] = np.log(df_calc[self.target_col]).diff() * 100
        
        # Rellenar el primer valor NaN de los retornos con 0 para no perder esa fila prematuramente
        df_calc['log_ret'] = df_calc['log_ret'].fillna(0)
        
        # Pre-llenamos con NaN los primeros 500 días (fase de calentamiento)
        forecasts = [np.nan] * self.window_size
        
        # 2. Rolling Window (Ciclo de pronóstico)
        for i in range(self.window_size, len(df_calc)):
            # Extraer la ventana de datos exacta para este día
            train_window = df_calc['log_ret'].iloc[i - self.window_size : i]
            
            # SALVAVIDAS 1: Calculamos la volatilidad simple histórica
            vol_historica = train_window.std()
            
            try:
                # Intento 1: EGARCH (Asimétrico, ideal pero matemáticamente inestable)
                model = arch_model(train_window, vol='EGarch', p=1, o=1, q=1, dist='t', rescale=False)
                res = model.fit(disp='off', show_warning=False)
                pred = res.forecast(horizon=1, reindex=False)
                vol_pred = np.sqrt(pred.variance.values[-1, 0])
                
                # FILTRO DE CORDURA: Si es NaN, infinito, o irrealmente alto (> 5% de volatilidad diaria)
                if np.isnan(vol_pred) or np.isinf(vol_pred) or vol_pred > 5.0:
                    
                    # Intento 2: GARCH(1,1) estándar (Simétrico, mucho más robusto)
                    model_fallback = arch_model(train_window, vol='Garch', p=1, q=1, rescale=False)
                    res_fallback = model_fallback.fit(disp='off', show_warning=False)
                    pred_fallback = res_fallback.forecast(horizon=1, reindex=False)
                    vol_pred = np.sqrt(pred_fallback.variance.values[-1, 0])
                    
                    # Intento 3: Si todo el álgebra lineal falla
                    if np.isnan(vol_pred) or np.isinf(vol_pred) or vol_pred > 5.0:
                        vol_pred = vol_historica
                        
            except Exception:
                # Si la librería arroja cualquier error fatal de convergencia
                vol_pred = vol_historica
                
            forecasts.append(vol_pred)
            
            # Imprimir progreso cada 200 días para no saturar la consola
            if i % 200 == 0:
                print(f"  > Procesando día {i}/{len(df_calc)}...")

        # 3. Asignación y Limpieza
        df_calc['EGARCH_Vol'] = forecasts
        
        # Eliminamos la columna auxiliar de retornos logarítmicos
        df_calc = df_calc.drop(columns=['log_ret'])
        
        # Eliminamos la fase de calentamiento (los primeros 500 días)
        df_calc = df_calc.dropna(subset=['EGARCH_Vol']).reset_index(drop=True)
        
        print("✅ EGARCH calculado y acoplado exitosamente.")
        return df_calc
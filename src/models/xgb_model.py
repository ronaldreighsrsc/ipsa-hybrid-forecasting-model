import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import joblib

warnings.filterwarnings("ignore")

class XGBoostTrainer:
    """
    Motor predictivo de Gradient Boosting (XGBoost).
    Implementa el pipeline corporativo de limpieza de Infs/NaNs, 
    clipping para control de outliers, Purged CV y Walk-Forward.
    """
    def __init__(self, look_back: int = 60, retrain_step: int = 50, 
                 n_splits: int = 3, purge_size: int = 60, embargo_size: int = 10):
        self.look_back = look_back
        self.retrain_step = retrain_step
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.scaler = StandardScaler()

    def prepare_sequences(self, df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:
        """
        Limpia desbordamientos (Infs) propios de los cálculos FFD y transforma 
        los datos bidimensionales en secuencias aplanadas para XGBoost.
        """
        # Parche anti-XGBoost Error: Limpiar Infs residuales antes del array
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        cols = [target_col] + [c for c in feature_cols if c != target_col]
        dataset = df_clean[cols].values
        
        X_raw, y_raw = [], []
        for i in range(self.look_back, len(dataset)):
            window = dataset[i-self.look_back:i, :]
            X_raw.append(window.flatten())
            # Etiqueta: 1 si sube, 0 si baja o se mantiene
            y_raw.append(1 if dataset[i, 0] > dataset[i-1, 0] else 0)
            
        return np.array(X_raw), np.array(y_raw)

    def _get_purged_embargoed_folds(self, num_samples: int) -> list:
        """Aplica la teoría de López de Prado para evitar Data Leakage temporal."""
        fold_size = num_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else num_samples
            train_indices = []
            for j in range(num_samples):
                if j < val_start - self.purge_size:          
                    train_indices.append(j)
                elif j >= val_end + self.embargo_size:       
                    train_indices.append(j)
            val_indices = list(range(val_start, val_end))
            folds.append((np.array(train_indices), np.array(val_indices)))
        return folds

    def find_best_params(self, X_train: np.ndarray, y_train: np.ndarray, param_grid: list) -> dict:
        """Búsqueda de la mejor arquitectura usando Purged CV."""
        print(f"  🔍 Buscando hiperparámetros (Purged & Embargoed CV para XGBoost)...")
        folds = self._get_purged_embargoed_folds(len(X_train))
        
        best_acc = -1
        best_params = param_grid[0]

        for params in param_grid:
            fold_accs = []
            for train_idx, val_idx in folds:
                if len(train_idx) == 0: continue
                
                model_cv = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric='logloss')
                model_cv.fit(X_train[train_idx], y_train[train_idx])
                preds = model_cv.predict(X_train[val_idx])
                fold_accs.append(accuracy_score(y_train[val_idx], preds))
            
            avg_acc = np.mean(fold_accs)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = params

        print(f"  ✅ Ganador: {best_params} (Acc Interno: {best_acc:.2%})")
        return best_params

    def walk_forward_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray, best_params: dict) -> tuple:
        """Entrena y predice paso a paso actualizando los pesos del árbol."""
        print(f"  🚀 Iniciando Walk-Forward (Reentrenamiento cada {self.retrain_step} días)...")
        
        # Escalamiento estricto y Clipping (Vital para estabilizar el gradiente de XGBoost)
        X_train_scaled = np.clip(self.scaler.fit_transform(X_train), -10, 10)
        X_test_scaled = np.clip(self.scaler.transform(X_test), -10, 10)

        # Entrenamiento Base
        master_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1, eval_metric='logloss')
        master_model.fit(X_train_scaled, y_train)

        pred_probs = []
        
        # Iteración Estep-by-Step
        for i in range(len(X_test_scaled)):
            prob = master_model.predict_proba(X_test_scaled[i].reshape(1, -1))[0][1]
            pred_probs.append(prob)
            
            # Reentrenamiento periódico
            if (i + 1) % self.retrain_step == 0:
                curr_X = np.concatenate((X_train_scaled, X_test_scaled[:i+1]))
                curr_y = np.concatenate((y_train, y_test[:i+1]))
                master_model.fit(curr_X, curr_y)

        # Extracción de importancia de variables agrupada por variable original
        num_features = X_train.shape[1] // self.look_back
        importances_flat = master_model.feature_importances_
        importances = importances_flat.reshape(self.look_back, num_features).sum(axis=0)
        
        return np.array(pred_probs), importances

    def save(self, filepath: str) -> None:
        """Saves the final trained model, scaler and params."""
        state = {
            'model': master_model if hasattr(self, '_master_model') else None,
            'scaler': self.scaler,
            'look_back': self.look_back
        }
        joblib.dump(state, filepath)
        print(f"  💾 XGBoost model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads a previously saved model."""
        state = joblib.load(filepath)
        instance = cls(look_back=state['look_back'])
        instance.scaler = state['scaler']
        instance._master_model = state['model']
        return instance
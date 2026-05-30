import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class RandomForestTrainer:
    """
    Motor de Machine Learning utilizando Random Forest.
    Implementa preparación de secuencias (Look-Back), escalamiento,
    validación cruzada avanzada (Purged & Embargoed) y predicción Walk-Forward.
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
        Transforma el DataFrame bidimensional en secuencias planas (ventanas de tiempo) 
        para que Random Forest pueda procesar memoria histórica.
        """
        # Aseguramos que el target esté en la posición 0 del array
        cols = [target_col] + [c for c in feature_cols if c != target_col]
        dataset = df[cols].values
        
        X_raw, y_raw = [], []
        for i in range(self.look_back, len(dataset)):
            window = dataset[i-self.look_back:i, :]
            X_raw.append(window.flatten())
            # Etiqueta: 1 si el precio subió respecto al día anterior, 0 si bajó/igual
            y_raw.append(1 if dataset[i, 0] > dataset[i-1, 0] else 0)
            
        return np.array(X_raw), np.array(y_raw)

    def _get_purged_embargoed_folds(self, num_samples: int) -> list:
        """Genera los índices para validación cruzada evitando Data Leakage."""
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
        """Ejecuta Grid Search usando Purged & Embargoed CV sobre los datos de entrenamiento."""
        print(f"  🔍 Buscando hiperparámetros (Purged & Embargoed CV)...")
        folds = self._get_purged_embargoed_folds(len(X_train))
        
        best_acc = -1
        best_params = param_grid[0]

        for params in param_grid:
            fold_accs = []
            for train_idx, val_idx in folds:
                model_cv = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
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
        """
        Escala los datos sin Leakage, entrena y predice paso a paso, 
        reentrenando periódicamente. Retorna probabilidades e importancia de variables.
        """
        print(f"  🚀 Iniciando Walk-Forward (Reentrenamiento cada {self.retrain_step} días)...")
        
        # 1. Escalamiento estricto (Fit solo en train, transform en test)
        X_train_scaled = np.clip(self.scaler.fit_transform(X_train), -10, 10)
        X_test_scaled = np.clip(self.scaler.transform(X_test), -10, 10)

        # 2. Modelo Base
        master_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        master_model.fit(X_train_scaled, y_train)

        pred_probs = []
        
        # 3. Iteración Paso a Paso
        for i in range(len(X_test_scaled)):
            prob = master_model.predict_proba(X_test_scaled[i].reshape(1, -1))[0][1]
            pred_probs.append(prob)
            
            # Reentrenamiento periódico
            if (i + 1) % self.retrain_step == 0:
                curr_X = np.concatenate((X_train_scaled, X_test_scaled[:i+1]))
                curr_y = np.concatenate((y_train, y_test[:i+1]))
                master_model.fit(curr_X, curr_y)

        # 4. Feature Importance final (del último modelo ajustado)
        # Calculamos cuántas variables originales hay (columnas totales / días de look_back)
        num_features = X_train.shape[1] // self.look_back
        importances_flat = master_model.feature_importances_
        
        # Agrupamos la importancia de los N días hacia atrás para cada variable original
        importances = importances_flat.reshape(self.look_back, num_features).sum(axis=0)
        
        return np.array(pred_probs), importances
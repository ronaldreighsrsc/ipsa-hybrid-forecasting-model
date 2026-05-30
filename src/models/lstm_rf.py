import os
import warnings

# Silenciar logs de C++ de TensorFlow ANTES de importarlo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class LSTMRFTrainer:
    """
    Motor Híbrido Secuencial (Deep Learning -> Machine Learning).
    Fase 1: Entrena un LSTM para entender patrones temporales en 3D.
    Fase 2: Corta la "cabeza" del LSTM y usa su última capa oculta como Feature Extractor.
    Fase 3: Alimenta estas nuevas variables (features aprendidos) a un Random Forest 
            para la clasificación final y el Walk-Forward.
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
        """Transforma el DataFrame bidimensional en Tensores 3D para el LSTM."""
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        cols = [target_col] + [c for c in feature_cols if c != target_col]
        dataset = df_clean[cols].values
        
        X_raw, y_raw = [], []
        for i in range(self.look_back, len(dataset)):
            X_raw.append(dataset[i-self.look_back:i, :])
            y_raw.append(1 if dataset[i, 0] > dataset[i-1, 0] else 0)
            
        return np.array(X_raw), np.array(y_raw)

    def _get_purged_embargoed_folds(self, num_samples: int) -> list:
        """Genera índices para validación cruzada evitando Data Leakage."""
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

    def _build_lstm_base(self, input_shape: tuple, units: int, dropout: float) -> Model:
        """Construye el modelo completo, asegurando un nombre específico para la capa extractora."""
        input_layer = Input(shape=input_shape)
        # Nombramos explícitamente esta capa para poder extraerla después
        lstm_layer = LSTM(units, return_sequences=False, name='lstm_extractor')(input_layer)
        dropout_layer = Dropout(dropout)(lstm_layer)
        output_layer = Dense(1, activation='sigmoid')(dropout_layer) 
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def find_best_params(self, X_train: np.ndarray, y_train: np.ndarray, lstm_grid: list) -> dict:
        """Grid Search 3D con K-Fold Purged enfocado en optimizar el extractor LSTM."""
        print(f"  🔍 [FASE 1] Buscando arquitectura óptima para el Extractor LSTM...")
        tf.keras.backend.clear_session()
        
        s_tr, t_steps, n_feat = X_train.shape
        temp_scaler = StandardScaler()
        X_train_scaled = np.clip(
            temp_scaler.fit_transform(X_train.reshape(-1, n_feat)), -5, 5
        ).reshape(s_tr, t_steps, n_feat)

        folds = self._get_purged_embargoed_folds(s_tr)
        best_acc = -1
        best_params = lstm_grid[0]

        for params in lstm_grid:
            fold_accs = []
            for train_idx, val_idx in folds:
                if len(train_idx) == 0: continue
                
                model_cv = self._build_lstm_base((t_steps, n_feat), units=params['units'], dropout=params['dropout'])
                es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
                
                model_cv.fit(X_train_scaled[train_idx], y_train[train_idx], epochs=10, batch_size=64, 
                             validation_data=(X_train_scaled[val_idx], y_train[val_idx]), 
                             callbacks=[es], verbose=0, shuffle=False)
                
                preds = (model_cv.predict(X_train_scaled[val_idx], verbose=0) > 0.5).astype(int)
                fold_accs.append(accuracy_score(y_train[val_idx], preds))
                
            avg_acc = np.mean(fold_accs) if fold_accs else 0
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = params

        print(f"    ✅ Ganador LSTM: {best_params} (Acc Interno: {best_acc:.2%})")
        return best_params

    def walk_forward_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray, best_params: dict) -> tuple:
        """
        1. Entrena el LSTM Maestro y extrae los features.
        2. Ejecuta el Walk-Forward actualizando EXCLUSIVAMENTE el Random Forest por eficiencia.
        """
        print(f"  🚀 [FASE 2] Generando Features y ejecutando Walk-Forward con Random Forest...")
        tf.keras.backend.clear_session()

        s_tr, t_steps, n_feat = X_train.shape
        s_te = X_test.shape[0]

        # Escalamiento Maestro
        X_train_scaled = np.clip(self.scaler.fit_transform(X_train.reshape(-1, n_feat)), -5, 5).reshape(s_tr, t_steps, n_feat)
        X_test_scaled = np.clip(self.scaler.transform(X_test.reshape(-1, n_feat)), -5, 5).reshape(s_te, t_steps, n_feat)

        # 1. Entrenar LSTM Base
        master_lstm = self._build_lstm_base((t_steps, n_feat), units=best_params['units'], dropout=best_params['dropout'])
        master_lstm.fit(X_train_scaled, y_train, epochs=15, batch_size=64, verbose=0, shuffle=False)

        # 2. Mutilar el LSTM para convertirlo en Extractor
        feature_extractor = Model(inputs=master_lstm.inputs, outputs=master_lstm.get_layer('lstm_extractor').output)
        
        # 3. Transformar 3D a 2D (Latent Features)
        X_train_features = feature_extractor.predict(X_train_scaled, verbose=0)
        X_test_features = feature_extractor.predict(X_test_scaled, verbose=0)

        # 4. Entrenar Random Forest Base sobre los features extraídos
        rf_model = RandomForestClassifier(n_estimators=150, max_depth=7, n_jobs=-1, random_state=42)
        rf_model.fit(X_train_features, y_train)

        pred_probs = []
        
        # 5. Iteración Walk-Forward (Solo re-entrenamos el RF)
        for i in range(len(X_test_features)):
            prob = rf_model.predict_proba(X_test_features[i].reshape(1, -1))[0][1]
            pred_probs.append(prob)
            
            # Re-entrenamiento periódico del clasificador
            if (i + 1) % self.retrain_step == 0:
                if (i+1) % (self.retrain_step*2) == 0:
                    print(f"    > Re-calibrando pesos del Random Forest en paso {i+1}/{len(X_test_features)}...")
                curr_X = np.concatenate((X_train_features, X_test_features[:i+1]))
                curr_y = np.concatenate((y_train, y_test[:i+1]))
                rf_model.fit(curr_X, curr_y)

        # Retornamos probabilidades. (La importancia de variables del RF aquí no es interpretable
        # directamente porque son "neuronas ocultas", por lo que devolvemos None).
        return np.array(pred_probs), None
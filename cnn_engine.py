# VolcanoAI/engines/cnn_engine.py
# -- coding: utf-8 --
"""
VOLCANO AI - CNN ENGINE (REVISED, RELU-ONLY)
===========================================
Versi ini mengunci fungsi aktivasi hidden layer menjadi ReLU saja
(permintaan client: "jangan pakai ELU, pakai ReLU saja").

Perubahan utama:
- Hanya 'relu' sebagai activation untuk semua hidden layers.
- Tuner (NNTuner) hanya mencoba 'relu' 
- Komentar / penjelasan tambahan mengenai alasan ReLU.
- Struktur fungsi (train, predict, export_results, manual_forward_pass) tetap dipertahankan.

Catatan teknis singkat tentang keputusan:
- ReLU (Rectified Linear Unit) adalah pilihan standar untuk hidden dense layer.
  Keunggulan: sederhana, efisien, mengurangi vanishing gradient, dan cepat konvergen.
- Karena model menggunakan BatchNormalization sebelum aktivasi, risiko "dying ReLU"
  berkurang (BN menstabilkan distribusi aktivasi).
"""

import os
import logging
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Gunakan tensorflow.keras untuk kompatibilitas
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup Logger
logger = logging.getLogger("VolcanoAI_CNN")
logger.addHandler(logging.NullHandler())

# =============================================================================
# SECTION 1: DATA PREPARATION (TABULAR FEATURE EXTRACTOR) - REVISED INPUT ORDER
# =============================================================================

class TabularFeatureExtractor:
    """
    Menyiapkan data tabular 5-Node Input sesuai spesifikasi:
    Node order (index):
      0 -> pusat ACO1    (aco_center_scalar)
      1 -> area ACO1     (aco_area_km2)
      2 -> pusat ACO2    (aco_center_prev)  # event previous (shift)
      3 -> area ACO2     (aco_area_prev)
      4 -> lstm_prediction (anomaly/score)

    Normalisasi: area dibagi norm_area; distance target dibagi norm_dist.
    """
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config
        # Normalisasi sederhana agar NN lebih cepat konvergen
        self.norm_area = float(self.cfg.get('norm_area', 1000.0))  # pembagi untuk area (km2)
        self.norm_dist = float(self.cfg.get('norm_dist', 100.0))   # pembagi untuk jarak (km)

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menghasilkan X (N x 5) dan Y (N x 2) untuk training.
        Target dihitung dari pergerakan koordinat aktual (bearing & haversine distance).
        Alignment: karena target menggunakan baris i -> i+1, kita drop baris terakhir pada X.
        """
        if df.empty:
            return np.array([]), np.array([])

        df = df.copy().sort_values('Acquired_Date').reset_index(drop=True)

        # --- Definisi ACO1 & ACO2 (ACO2 = previous event) ---
        df['aco_center_scalar'] = df.get('aco_center_lat', 0.0).fillna(0.0)
        df['aco_area_prev'] = df['aco_area_km2'].shift(1).fillna(0.0) #(t-1)
        df['aco_center_prev'] = df['aco_center_scalar'].shift(1).fillna(0.0)

        if 'lstm_prediction' not in df.columns:
            df['lstm_prediction'] = df.get('anomaly_score', df.get('PheromoneScore', 0.0)).fillna(0.0)

        feature_cols = [ #Rumus 3.36, 3.37, 3.38, input cnn
            'aco_center_scalar',  # pusat ACO1
            'aco_area_km2',       # area ACO1
            'aco_center_prev',    # pusat ACO2 (previous)
            'aco_area_prev',      # area ACO2 (previous)
            'lstm_prediction'     # output LSTM (anomali)
        ]

        X = df[feature_cols].fillna(0.0).values.astype(float)

        # Normalisasi: area columns (index 1 and 3)
        if X.shape[0] > 0:
            X[:, 1] /= self.norm_area
            X[:, 3] /= self.norm_area

        # --- TARGET CALCULATION (Y: bearing_deg, distance_km) ---
        lat = df['EQ_Lintang'].values
        lon = df['EQ_Bujur'].values

        if len(lat) < 2:
            return np.array([]), np.array([])

        lat1 = lat[:-1]
        lon1 = lon[:-1]
        lat2 = lat[1:]
        lon2 = lon[1:]

        def calculate_bearing_vec(lat1, lon1, lat2, lon2):
            lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
            dlon_rad = np.radians(lon2 - lon1)
            y = np.sin(dlon_rad) * np.cos(lat2_rad)
            x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
            bearing = np.degrees(np.arctan2(y, x))
            return (bearing + 360) % 360

        def calculate_distance_vec(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c

        target_angles = calculate_bearing_vec(lat1, lon1, lat2, lon2)
        target_dists = calculate_distance_vec(lat1, lon1, lat2, lon2)

        X_final = X[:-1]
        Y_final = np.column_stack((target_angles, target_dists)).astype(float) #Rumus 3.42

        valid_mask = ~np.isnan(Y_final).any(axis=1)
        X_final = X_final[valid_mask]
        Y_final = Y_final[valid_mask]

        # Normalisasi target to [0,1] for network convenience
        Y_final[:, 0] = Y_final[:, 0] / 360.0       # angle scaled to [0,1] Rumus 3.43
        Y_final[:, 1] = Y_final[:, 1] / self.norm_dist

        return X_final, Y_final

    def denormalize_output(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Mengembalikan output dari skala [0,1] ke skala fisik (degree, km)
        """
        y_real = np.zeros_like(y_pred)
        y_real[:, 0] = (y_pred[:, 0] * 360.0) % 360.0
        y_real[:, 1] = y_pred[:, 1] * self.norm_dist
        return y_real

# =============================================================================
# SECTION 2: NEURAL NETWORK ARCHITECTURE (SIMPLE, 2-3 HIDDEN LAYERS, RELU-ONLY)
# =============================================================================

class SimpleNNFactory: #Rumus 3.39, 3.40, 3.41
    """
    Bangun model Dense sederhana dengan 2 atau 3 hidden layer.
    **Catatan penting**: Aktivasi hidden layer dikunci menjadi 'relu' (ReLU-only).
    Keterangan mengapa ReLU:
      - ReLU sederhana dan efisien.
      - Mengurangi risiko vanishing gradient.
      - Kompatibel dengan BatchNormalization (yang kita pakai).
      - Sesuai untuk regresi dengan input yang telah dinormalisasi.
    """
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config

    def build_model(self, params: Dict[str, Any] = None) -> Model:
        p = params if params else {}
        # Default nodes kecil
        hidden_count = int(p.get('hidden_count', 2))  # 2
        units = p.get('units', [32, 16, 8])
        dropout = float(p.get('dropout', 0.0))
        lr = float(p.get('learning_rate', 0.001))

        # PAKAI HANYA RELU (override jika ada p['activation'])
        activation = 'relu' 

        # Pastikan units list cukup panjang untuk hidden_count
        if len(units) < hidden_count:
            units = units + [8] * (hidden_count - len(units))

        model = Sequential(name="Simple_CNN_VolcanoAI_Revised_RELU_ONLY")

        # Layer 1 — INPUT LAYER
        model.add(Input(shape=(5,), name="Input_5_Nodes")) 

        # Hidden Layers (loop)
        for i in range(hidden_count):
            model.add(Dense(units[i], name=f"Hidden_{i+1}"))
            # BatchNormalization membantu stabilitas dan mengurangi masalah aktivasi
            model.add(BatchNormalization())
            model.add(Activation(activation))  
            if dropout > 0:
                model.add(Dropout(dropout))
        """
        *Layer 2 — Dense Hidden Layer 1*

        Bobot = 5 × 8 = 40
        Bias  = 8
        Total parameter = 48

        *Layer 3 — BatchNormalization (Hidden 1)*
        -Menstabilkan distribusi output dari Dense
        -tidak ada neuron, bobot, dan bias.

        *Layer 4 — Activation ReLU (Hidden 1)*
        -tidak ada neuron, bobot, dan bias.

        *Layer 5 — Dense Hidden Layer 2*

        Bobot = 8 × 4 = 32
        Bias  = 4
        Total = 36

        (untuk BatchNormalization dan Activation ReLU sama (tidak ada neuron, bobot, dan bias.))

        """

        # Output Layer
        # Memiliki bobot & bias 
        # Aktivasi: linear
        model.add(Dense(2, activation='linear', name="Output_2_Nodes")) # Jumlah node: 2

        """
        Bobot = 4 × 2 = 8
        Bias  = 2
        Total = 10
        """

        
        model.compile(
            optimizer=Adam(learning_rate=lr), #Rumus 3.44
            loss='mse',   # regression MSE
            metrics=['mae']
        )

        return model
        
        """
        Jadi:
        - total layer 8                     - Input     : 5
        - Dense layer (belajar) 3           - Hidden 1  : 8
        - Hidden layer (Dense) 2            - Hidden 2  : 4
        - Output layer 1                    - Output    : 2
        """
# =============================================================================
# SECTION 3: TUNER & ENGINE (KOMPATIBILITAS DENGAN CODE LAMA)
# =============================================================================

class NNTuner:
    """
    Tuner sederhana. Opsi activation hanya 'relu'
    """
    def __init__(self, factory: SimpleNNFactory, trials=3):
        self.factory = factory
        self.trials = trials
        # Grid kecil; activation hanya 'relu' 
        self.grid = {
            'hidden_count': [2, 3],
            'units': [[32,16], [32,16,8], [16,8]],
            'learning_rate': [0.001, 0.005],
            'activation': ['relu']  
        }

    def _sample_params(self):
        params = {k: random.choice(v) for k, v in self.grid.items()}
        if isinstance(params['units'], list) and len(params['units']) < params['hidden_count']:
            params['units'] = params['units'] + [8] * (params['hidden_count'] - len(params['units']))
        return params

    def search(self, X_train, Y_train, X_val, Y_val) -> Dict[str, Any]:
        best_loss = float('inf')
        best_params = {'hidden_count': 2, 'units': [32,16], 'learning_rate': 0.001, 'activation':'relu'}
        logger.info(f" [NN Tuner] Memulai {self.trials} trial optimasi (ReLU-only)...")
        for i in range(self.trials):
            params = self._sample_params()
            K.clear_session()
            try:
                # Note: SimpleNNFactory akan mengabaikan params['activation'] dan memakai 'relu'
                model = self.factory.build_model(params)
                hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=16, verbose=0)
                val_loss = hist.history['val_loss'][-1]
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
            except Exception as e:
                logger.warning(f"Tuner trial failed: {e}")
                continue
        logger.info(f" [NN Tuner] Params terbaik (ReLU-only): {best_params}")
        return best_params

class CnnEngine:
    """
    Engine utama: training, predict, export_results, evaluate, dsb.
    Struktur API sama seperti versi sebelumnya.
    """
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config
        self.cfg.setdefault('epochs', 50)
        self.cfg.setdefault('batch_size', 16)
        # apakah akan dedup input_summary ketika export
        self.cfg.setdefault('dedup_input_summary', True)

        self.model_dir = Path(self.cfg.get('model_dir', 'output/cnn/models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = Path(self.cfg.get('output_dir', 'output/cnn_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = TabularFeatureExtractor(self.cfg)
        self.factory = SimpleNNFactory(self.cfg)
        self.tuner = NNTuner(self.factory)
        self.models = {}

    def train(self, df_train: pd.DataFrame, lstm_engine=None) -> bool:
        if df_train.empty: return False
        logger.info("=== START TRAINING SIMPLE NN (5-INPUT / 2-OUTPUT) REVISED (RELU-ONLY) ===")
        unique_clusters = sorted([c for c in df_train['cluster_id'].unique() if c != -1])
        success_count = 0
        for cid in unique_clusters:
            logger.info(f">>> Training Cluster {cid}")
            df_c = df_train[df_train['cluster_id'] == cid]
            X, Y = self.extractor.prepare_dataset(df_c)
            if len(X) < 10:
                logger.warning(f"Skip c{cid}: Data kurang ({len(X)} sampel).")
                continue
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            Y_train, Y_val = Y[:split], Y[split:]
            params = self.tuner.search(X_train, Y_train, X_val, Y_val)
            model = self.factory.build_model(params)
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=self.model_dir / f"cnn_model_c{cid}.keras", save_best_only=True)
            ]
            hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                             epochs=self.cfg['epochs'], batch_size=self.cfg['batch_size'],
                             callbacks=callbacks, verbose=1)
            self.models[cid] = model
            success_count += 1
            loss = hist.history['loss'][-1]
            logger.info(f"Cluster {cid} Trained. Final Loss: {loss:.4f}")
            try:
                summary = self._model_weight_bias_summary(model)
                logger.info(f"Model params summary c{cid}: {summary}")
            except Exception:
                pass
        return success_count > 0

    def predict(self, df_predict: pd.DataFrame, lstm_engine=None) -> pd.DataFrame:
        df_out = df_predict.copy()
        for col in [
            'cnn_angle_deg',
            'cnn_distance_km',
            'cnn_confidence',
            'cnn_cardinal',
            'cnn_dx_km',
            'cnn_dy_km',
            'cnn_direction_text'
        ]:
            df_out[col] = np.nan

        unique_clusters = sorted([c for c in df_out['cluster_id'].unique() if c != -1])

        for cid in unique_clusters:
            model = self.models.get(cid)
            if not model:
                try:
                    path = self.model_dir / f"cnn_model_c{cid}.keras"
                    if path.exists():
                        model = load_model(path, compile=False)
                        self.models[cid] = model
                except Exception:
                    pass
            if not model:
                logger.warning(f"No model for c{cid}, skipping prediction.")
                continue

            mask = df_out['cluster_id'] == cid
            df_c = df_out[mask]
            if df_c.empty: continue

            # Prepare input same as extractor but using only available rows
            df_c_proc = df_c.copy().sort_values('Acquired_Date')
            df_c_proc['aco_area_prev'] = df_c_proc['aco_area_km2'].shift(1).fillna(0.0)
            df_c_proc['aco_center_scalar'] = df_c_proc.get('aco_center_lat', 0.0).fillna(0.0)
            df_c_proc['aco_center_prev'] = df_c_proc['aco_center_scalar'].shift(1).fillna(0.0)
            if 'lstm_prediction' not in df_c_proc.columns:
                df_c_proc['lstm_prediction'] = df_c_proc.get('anomaly_score', df_c_proc.get('PheromoneScore', 0.0)).fillna(0.0)

            cols = ['aco_center_scalar','aco_area_km2','aco_center_prev','aco_area_prev','lstm_prediction']
            X = df_c_proc[cols].fillna(0.0).values.astype(float)
            if X.shape[0] > 0:
                X[:,1] /= self.extractor.norm_area
                X[:,3] /= self.extractor.norm_area

            try:
                preds_norm = model.predict(X, verbose=0)
                preds_real = self.extractor.denormalize_output(preds_norm)
                angles = np.abs(preds_real[:,0]) % 360.0
                dists = np.abs(preds_real[:,1])
                idx_aligned = df_c_proc.index

                cardinals = []
                dx_list, dy_list, text_list = [], [], []

                for a, d in zip(angles, dists):
                    card = self._get_cardinal(a)
                    dx, dy = self._bearing_to_vector(a, d)

                    cardinals.append(card)
                    dx_list.append(dx)
                    dy_list.append(dy)
                    text_list.append(f"Bergerak ke arah {card} sejauh {d:.2f} km")

                df_out.loc[idx_aligned, 'cnn_angle_deg'] = angles
                df_out.loc[idx_aligned, 'cnn_distance_km'] = dists
                df_out.loc[idx_aligned, 'cnn_confidence'] = 0.85
                df_out.loc[idx_aligned, 'cnn_cardinal'] = cardinals
                df_out.loc[idx_aligned, 'cnn_dx_km'] = dx_list
                df_out.loc[idx_aligned, 'cnn_dy_km'] = dy_list
                df_out.loc[idx_aligned, 'cnn_direction_text'] = text_list

            except Exception as e:
                logger.error(f"Prediction error c{cid}: {e}")

        return df_out

    def predict_and_export( #Rumus 3.45, 3.46, 3.47
        self,
        df_predict: pd.DataFrame,
        lstm_engine=None,
        filename: str | None = None
    ) -> pd.DataFrame:
        """
        Melakukan inferensi CNN dan export hasil ke Excel (1 kali saja).
        """

        # 1️⃣ Inferensi
        df_out = self.predict(df_predict, lstm_engine=lstm_engine)

        # 2️⃣ Meta info
        meta = {
            "engine": "CNN",
            "model_type": "Simple NN (5 Input / 2 Output) Revised (ReLU-only)",
            "rows_input": int(len(df_predict)),
            "rows_output": int(len(df_out)),
            "generated_at": datetime.now().isoformat()
        }

        # 3️⃣ Export
        self.export_results(
            df_input=df_predict,
            df_output=df_out,
            meta=meta,
            filename=filename
        )

        return df_out


    def _get_cardinal(self, angle):
        dirs = ["Utara", "Timur Laut", "Timur", "Tenggara", "Selatan", "Barat Daya", "Barat", "Barat Laut"]
        ix = int(round(angle / (360. / len(dirs))))
        return dirs[ix % len(dirs)]

    def _bearing_to_vector(self, angle_deg: float, distance_km: float):
        """
        Konversi bearing + jarak menjadi vektor arah (dx, dy)
        dx: positif = Timur, negatif = Barat
        dy: positif = Utara, negatif = Selatan
        """
        rad = np.radians(angle_deg)
        dx = distance_km * np.sin(rad) #Rumus 3.48
        dy = distance_km * np.cos(rad) #Rumus 3.49
        return dx, dy


    # -------------------
    # Utility: weight & bias summary and manual forward
    # -------------------
    def _model_weight_bias_summary(self, model: Model) -> Dict[str, Any]:
        """
        Mengembalikan ringkasan jumlah bobot & bias per Dense layer.
        Rumus jumlah bobot untuk Dense: W.shape = (in_dim, out_units), b.shape = (out_units,)
        jumlah_params = in_dim * out_units + out_units
        """
        summary = {}
        total = 0
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                w, b = layer.get_weights()
                in_dim, out_units = w.shape
                params = in_dim * out_units + out_units
                summary[layer.name] = {
                    'weights_shape': w.shape,
                    'bias_shape': b.shape,
                    'params_count': int(params)
                }
                total += params
        summary['total_params'] = int(total)
        return summary

    def manual_forward_pass(self, model: Model, x_input: np.ndarray, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Contoh perhitungan numeric forward pass (manual) dari input -> setiap hidden -> output.
        - x_input: shape (n_features,) atau (1, n_features)
        - Mengambil bobot & bias dari model.get_weights() layer by layer

        Formula per layer (Dense):
          z = W^T x + b    (jika W diberi bentuk (in_dim, out_units) dan x vektor kolom)
          a = activation(z)

        NOTE: BatchNorm diabaikan dalam perhitungan manual ini (untuk ringkasan).
        Activation yang di-support: relu, tanh, sigmoid (relu adalah yang dipakai di model ini).
        """
        x = x_input.reshape(-1) if x_input.ndim > 1 else x_input
        activations = {}
        curr = x.astype(float)

        layer_idx = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                w, b = layer.get_weights()
                z = np.dot(curr, w) + b
                act_name = 'linear'
                next_index = layer_idx + 1
                if next_index < len(model.layers) and isinstance(model.layers[next_index], tf.keras.layers.Activation):
                    act_name = model.layers[next_index].activation.__name__
                else:
                    if hasattr(layer, 'activation'):
                        try:
                            act_name = layer.activation.__name__
                        except Exception:
                            act_name = 'linear'
                # apply activation (support common ones)
                if act_name in ('relu', 'elu', 'tanh', 'sigmoid'):
                    if act_name == 'relu':
                        a = np.maximum(0, z)
                    elif act_name == 'elu':
                        # Note: ELU won't be used in this RE+LU-only build, but support kept for completeness
                        a = np.where(z > 0, z, np.expm1(z))
                    elif act_name == 'tanh':
                        a = np.tanh(z)
                    elif act_name == 'sigmoid':
                        a = 1 / (1 + np.exp(-z))
                else:
                    a = z

                activations[layer.name] = {
                    'z': z,
                    'a': a,
                    'activation': act_name,
                    'weights_shape': w.shape,
                    'bias_shape': b.shape
                }
                curr = a
            layer_idx += 1

        if verbose:
            print("--- Manual forward pass detail ---")
            for lname, info in activations.items():
                print(f"Layer {lname}: weights_shape={info['weights_shape']}, bias_shape={info['bias_shape']}")
                print(f" z (first 5) = {np.round(info['z'][:5],6)}")
                print(f" a (first 5) = {np.round(info['a'][:5],6)}")
            print("--- End manual forward ---")

        return activations

    # =========================================================
    # EXPORT & EVALUATION (tidak banyak berubah)
    # =========================================================
    def export_results(
        self,
        df_input: pd.DataFrame,
        df_output: pd.DataFrame,
        meta: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Export hasil CNN ke Excel multi-sheet.
        - Sheet utama: CNN_Presentation (untuk client)
        - Sheet Meta
        - Sheet Raw_Output (untuk engineer)
        """

        try:
            # =========================
            # OUTPUT PATH
            # =========================
            out_dir = Path(self.results_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = filename if filename else f"cnn_presentation_{ts}.xlsx"
            out_path = out_dir / fname

            # =========================
            # CLIENT PRESENTATION DATA
            # (HARUS dari df_output)
            # =========================
            present_cols = [
                "Acquired_Date",      # waktu, output cnn
                "cnn_angle_deg",      # sudut
                "cnn_cardinal",       # arah
                "cnn_distance_km",    # jarak
            ]
            present_cols = [c for c in present_cols if c in df_output.columns]

            if present_cols:
                output_summary = df_output[present_cols].copy()

                if "Acquired_Date" in output_summary.columns:
                    output_summary["Acquired_Date"] = (
                        output_summary["Acquired_Date"].astype(str)
                    )
            else:
                output_summary = pd.DataFrame(columns=present_cols)


            # =========================
            # META INFO
            # =========================
            meta = meta or {}
            meta.setdefault("engine", "CNN")
            meta.setdefault("model_type", "Simple NN (5 Input / 2 Output) ReLU-only")
            meta.setdefault("rows_input", int(len(df_input)))
            meta.setdefault("rows_output", int(len(df_output)))
            meta.setdefault("generated_at", datetime.now().isoformat())
            meta.setdefault(
                "notes",
                "CNN presentation export (prediction-based, client-ready)"
            )

            # =========================
            # WRITE EXCEL (MULTI-SHEET)
            # =========================
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

                # 1️⃣ SHEET UTAMA (CLIENT)
                output_summary.to_excel(
                    writer,
                    sheet_name="CNN_Presentation",
                    index=False
                )

                # 2️⃣ META
                pd.DataFrame([meta]).to_excel(
                    writer,
                    sheet_name="Meta",
                    index=False
                )

                # 3️⃣ RAW OUTPUT (ENGINEER)
                try:
                    df_output.to_excel(
                        writer,
                        sheet_name="Raw_Output",
                        index=False
                    )
                except Exception:
                    pd.DataFrame(df_output).to_excel(
                        writer,
                        sheet_name="Raw_Output",
                        index=False
                    )

            logger.info(f"[CNN] Presentation Excel saved → {out_path}")
            return out_path

        except Exception as e:
            logger.error("[CNN] export_results failed", exc_info=True)
            return None


    def evaluate_predictions(self, df_out: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
        df = df_out.copy()
        df['cnn_correct'] = False
        if 'cnn_distance_km' in df and 'ga_distance_km' in df:
            df['dist_err'] = (df['cnn_distance_km'] - df['ga_distance_km']).abs()
        if 'cnn_angle_deg' in df and 'ga_bearing_deg' in df:
            df['angle_err'] = (df['cnn_angle_deg'] - df['ga_bearing_deg']).abs() % 360
            df['angle_err'] = df['angle_err'].apply(lambda x: 360-x if x>180 else x)
        cond = pd.Series(True, index=df.index)
        if 'dist_km' in thresholds and 'dist_err' in df:
            cond &= df['dist_err'] <= thresholds['dist_km']
        if 'angle_deg' in thresholds and 'angle_err' in df:
            cond &= df['angle_err'] <= thresholds['angle_deg']
        df['cnn_correct'] = cond
        return df

# =============================================================================
# EXAMPLE: Cara menggunakan fungsi manual_forward_pass untuk melihat perhitungan bobot/bias
# =============================================================================
if __name__ == "__main__":
    cfg = {'norm_area':1000.0,'norm_dist':100.0}
    factory = SimpleNNFactory(cfg)
    params = {'hidden_count':2, 'units':[8,4], 'learning_rate':0.001}  # hidden berjumlah 2 dengan node/unit 8,4 = total 2 layer
    model = factory.build_model(params)

    x_sample = np.array([0.1, 0.05, 0.08, 0.02, 0.0])


    print(model.summary())
        

    engine = CnnEngine(cfg)
    summary = engine._model_weight_bias_summary(model)
    print('weight/bias summary:', summary)

    activs = engine.manual_forward_pass(model, x_sample)


    print('Done demo')

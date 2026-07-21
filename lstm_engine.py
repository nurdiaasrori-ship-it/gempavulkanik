# VolcanoAI/engines/lstm_engine.py
# -- coding: utf-8 --

"""
VOLCANO AI - LSTM TITAN ENGINE V6.2 (THE NEURAL CORE ULTIMATE)
==============================================================
Modul ini adalah Jantung Utama dari sistem prediksi VolcanoAI.
Mengimplementasikan arsitektur Deep Learning Hybrid yang menggabungkan:
1.  Sequence Modeling (Bi-LSTM)
2.  Attention Mechanism (Bahdanau)
3.  Probabilistic Forecasting (Gaussian NLL)
4.  Realtime Buffer Management (Sliding Window)
5.  Anomaly Detection System (Z-Score & Drift Monitoring)
6.  State Persistence (untuk integrasi CNN)

Copyright (c) 2025 VolcanoAI Team.
"""

import os # untuk operasi file dan direktori
import sys # untuk manipulasi path sistem
import time # untuk pengukuran waktu eksekusi
import json # untuk serialisasi metadata
import math # untuk fungsi matematika dasar
import shutil # untuk operasi file tingkat lanjut
import random # untuk sampling acak dalam hyperparameter tuning
import logging # untuk logging sistem
import pickle # untuk serialisasi objek
import functools # untuk dekorator fungsi
import uuid # untuk pembuatan UUID unik
from pathlib import Path # untuk manipulasi path yang lebih baik
from datetime import datetime # untuk operasi tanggal dan waktu
from typing import Any, Dict, List, Optional, Tuple, Union # untuk anotasi tipe
# Data Science Libs
import numpy as np # untuk operasi array dan tensor
import pandas as pd # untuk manipulasi data tabular
import matplotlib # untuk visualisasi data
matplotlib.use("Agg") # Backend non-interaktif untuk server
import matplotlib.pyplot as plt # untuk plotting
import seaborn as sns # untuk visualisasi statistik

# Machine Learning Libs
from joblib import dump, load # untuk serialisasi model dan scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # untuk scaling fitur
from sklearn.cluster import DBSCAN # untuk clustering spasial
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # untuk evaluasi model
from scipy.spatial.distance import mahalanobis # untuk perhitungan jarak Mahalanobis
from sklearn.impute import KNNImputer # untuk imputasi data hilang

# Deep Learning Libs (TensorFlow/Keras)
import tensorflow as tf # untuk operasi tensor dan model DL
from keras import backend as K # untuk fungsi backend Keras
from keras.models import Model, load_model # untuk definisi dan pemuatan model
from keras.layers import (
    Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, 
    Bidirectional, Concatenate, Conv1D, GlobalAveragePooling1D, 
    Layer, Dot, Activation, BatchNormalization, Add, Multiply, 
    Lambda, Reshape, Permute, Flatten, GaussianNoise
) # untuk lapisan neural network
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    CSVLogger, TensorBoard, LearningRateScheduler
) # untuk callback pelatihan
from keras.optimizers import Adam, RMSprop # untuk optimizers

# Config Imports (Safe Loader)
try:
    from ..config.lstm_config import LstmPipelineConfig # Konfigurasi Pipeline LSTM
except ImportError:
    pass

try:
    from ..processing.feature_engineer import FeatureEngineer # Feature Engineering Module
    from ..config.config import CONFIG # Global Config
except ImportError:
    pass

# Setup Logger
logger = logging.getLogger("VolcanoAI.LstmEngine") # Nama logger spesifik
logger.addHandler(logging.NullHandler()) # Hindari duplikasi handler

# =============================================================================
# SECTION 1: MATH KERNEL & UTILITIES (THE FOUNDATION)
# =============================================================================
# Modul matematika kustom untuk operasi tensor tingkat rendah
def execution_telemetry(func):
    """Decorator untuk memantau kinerja setiap fungsi kritis."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            t1 = time.perf_counter()
            # logger.debug(f"[Telemetry] {func.__name__} executed in {t1-t0:.4f}s")
    return wrapper
# class untuk mengelompokkan fungsi matematika kustom
class MathKernel:
    """
    Kernel matematika kustom untuk operasi tensor tingkat rendah.
    Menangani fungsi kerugian probabilistik (Probabilistic Loss Functions).
    """
    @staticmethod
    def gaussian_nll(y_true, y_pred): # fungsi untuk Negative Log-Likelihood Gaussian
        # Pisahkan output model: [Mean, Variance]
        mu = y_pred[..., 0:1]
        sigma = y_pred[..., 1:2]
        
        # [FIX KRITIS]: Clip sigma dengan batas bawah yang sangat aman
        # Batas atas 1e6 juga penting untuk mencegah NaN karena log(Inf)
        sigma = tf.clip_by_value(sigma, 1e-5, 1e6) 
        
        # Gunakan tf.math.log dan tf.math.square untuk keamanan di TF 2.x
        # NLL = 0.5 * log(sigma) + 0.5 * (y - mu)^2 / sigma
        nll = 0.5 * tf.math.log(sigma) + 0.5 * tf.math.square(y_true - mu) / sigma
        return tf.reduce_mean(nll)

    @staticmethod
    def uncertainty_metric(y_true, y_pred): # fungsi metrik ketidakpastian
        """Metrik pemantau: Rata-rata ketidakpastian (sigma) yang diprediksi model."""
        sigma = y_pred[..., 1:2]
        # [FIX] Clip sigma untuk mencegah nilai ekstrem negatif
        return tf.reduce_mean(tf.clip_by_value(sigma, 1e-6, 1e6))

    @staticmethod
    def mean_absolute_error_mu(y_true, y_pred): # fungsi MAE khusus
        """[FIX KRITIS] Metrik MAE yang hanya membandingkan Y_true dengan Mean (kolom 0) dari Y_pred."""
        mu = y_pred[..., 0:1] # Ambil hanya kolom Mean (Mu)
        return tf.reduce_mean(tf.abs(y_true - mu))

    @staticmethod
    def calculate_z_score(value, mean, std): # fungsi Z-Score
        """Hitung Z-Score untuk deteksi outlier standar."""
        if std < 1e-9: return 0.0
        return (value - mean) / std

# =============================================================================
# SECTION 2: DATA MANAGEMENT & CLUSTERING (THE ORGANIZER)
# =============================================================================
# class untuk menjaga integritas data
class DataGuard:
    """Penjaga integritas data sebelum masuk ke neural network."""
    def __init__(self, required_columns: List[str]): # kolom wajib
        self.required_columns = required_columns
    # fungsi validasi struktur data
    def validate_structure(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty: return False
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing: return False
        return True
    # fungsi sanitasi kolom temporal
    def sanitize_temporal(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        df_clean = df.copy()
        if date_col in df_clean.columns:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[date_col])
            # PENTING: Reset index agar urut 0..N untuk slicing tensor
            df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
        return df_clean
    # fungsi sanitasi kolom numerik
    def sanitize_numeric(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        return df
# class untuk mengelompokkan data gempa berdasarkan lokasi
class GeoClusterer:
    """
    Mengelompokkan gempa berdasarkan kedekatan spasial.
    Setiap cluster (misal: Gunung Semeru, Gunung Raung) akan punya 'Otak' (Model) sendiri.
    """
    def __init__(self, eps: float, min_samples: int, metric: str = "haversine"): # inisialisasi DBSCAN
        self.eps = eps # radius epsilon
        self.min_samples = min_samples # jumlah minimum sampel
        self.metric = metric # metrik jarak
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric) # model DBSCAN
    # fungsi fit dan prediksi cluster
    def fit_predict(self, df: pd.DataFrame, lat_col="EQ_Lintang", lon_col="EQ_Bujur") -> pd.Series:
        if lat_col not in df.columns or lon_col not in df.columns: # jika kolom tidak ada
            return pd.Series([-1] * len(df), index=df.index, name='cluster_id') # kembalikan -1 semua
        
        valid = df[[lat_col, lon_col]].dropna() # ambil data valid
        if valid.empty: return pd.Series([-1]*len(df), index=df.index, name='cluster_id') # jika kosong kembalikan -1 semua 
        
        # DBSCAN Haversine butuh radian
        coords = np.radians(valid.values) # konversi ke radian
        labels = self.model.fit_predict(coords) # fit dan prediksi
        # Buat Series lengkap dengan -1 untuk data invalid
        full_labels = pd.Series(-1, index=df.index, name='cluster_id') # inisialisasi -1
        full_labels.loc[valid.index] = labels # isi label valid
        # Logging jumlah cluster teridentifikasi
        n_c = len(set(labels)) - (1 if -1 in labels else 0) # hitung cluster valid
        logger.info(f"[GeoClusterer] Teridentifikasi {n_c} cluster spasial aktif.") # log info
        return full_labels

# =============================================================================
# SECTION 3: TENSOR FACTORY (SEQUENCE GENERATION)
# =============================================================================
# class untuk membuat tensor input/output untuk LSTM
class TensorFactory:
    """
    Pabrik Tensor: Mengubah data tabular 2D menjadi Array 3D [Samples, TimeSteps, Features].
    Menangani logika Sliding Window dan Teacher Forcing untuk arsitektur Seq2Seq.
    """
    def __init__(self, features: List[str], target: str, seq_len: int, pred_len: int): # inisialisasi pabrik tensor
        self.features = features # daftar fitur
        self.target = target # target prediksi
        self.seq_len = seq_len # panjang urutan input
        self.pred_len = pred_len # panjang urutan prediksi
        
        # Pastikan target ada dalam daftar fitur
        if target not in features: # jika target tidak ada
            self.features.append(target) # tambahkan target ke fitur
        self.target_idx = self.features.index(target) # indeks target
        self.num_features = len(self.features) # jumlah fitur total
    # fungsi membuat tensor untuk training
    def construct_training_tensors(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Membuat tensor untuk Training Sequence-to-Sequence (Encoder-Decoder).
        
        Args:
            data (np.ndarray): Dataset (sudah diskalakan) dengan shape (n_rows, n_features)
            
        Returns:
            X_encoder (Batch, Seq_Len, Features)
            X_decoder (Batch, Pred_Len, Features) -> Digunakan untuk Teacher Forcing
            Y_target  (Batch, Pred_Len, 1)        -> Target Prediksi
        """
        n_rows = len(data) # jumlah baris data
        
        # Window = (Input Sequence) + (Prediction Horizon)
        window_size = self.seq_len + self.pred_len
        
        # [FIX]: Menggunakan n_rows < window_size.
        # Artinya jika data == window_size, kita masih bisa ambil 1 sampel.
        if n_rows < window_size:
            # Mengembalikan array kosong dengan shape yang benar agar code di bawahnya tidak crash
            return (
                np.zeros((0, self.seq_len, self.num_features)), 
                np.zeros((0, self.pred_len, self.num_features)), 
                np.zeros((0, self.pred_len, 1))
            )
        
        X_enc_list, X_dec_list, Y_list = [], [], []
        
        # Iterasi Sliding Window
        # Range berhenti di: Total - Window + 1 agar indeks terakhir tercakup
        for i in range(n_rows - window_size + 1):
            
            # --- Indices ---
            idx_start_enc = i
            idx_end_enc   = i + self.seq_len  # Batas antara encoder dan prediksi
            
            idx_start_pred = idx_end_enc
            idx_end_pred   = idx_start_pred + self.pred_len
            
            # --- Slicing ---
            
            # 1. Encoder Input: dari t=0 s/d t=seq_len
            x_enc_sample = data[idx_start_enc : idx_end_enc, :] #(t-1)
            
            # 2. Decoder Input (Teacher Forcing):
            #    Biasanya berupa Lag-1 dari target window.
            #    Kita ambil data dari akhir encoder (-1) sampai sebelum prediksi berakhir (-1).
            x_dec_sample = data[idx_end_enc - 1 : idx_end_pred - 1, :] #(t)
            
            # 3. Target Output (Y):
            #    Ambil hanya kolom target (misal: PheromoneScore).
            #    Asumsi: Target kolom terakhir (-1) atau sesuaikan indeksnya.
            #    Menggunakan slice data[start:end, -1:] agar dimensi tetap (Pred_Len, 1)
            target_col_idx = self.target_idx 
            
            # Menggunakan slice target_col_idx:target_col_idx+1 untuk menjaga dimensi (Pred_Len, 1)
            y_sample = data[idx_start_pred : idx_end_pred, target_col_idx:target_col_idx+1]
            
            X_enc_list.append(x_enc_sample)
            X_dec_list.append(x_dec_sample)
            Y_list.append(y_sample)
            
        return np.array(X_enc_list), np.array(X_dec_list), np.array(Y_list)
    # fungsi membuat tensor untuk inferensi
    def construct_inference_tensor(self, data: np.ndarray) -> np.ndarray:
        """Membuat tensor X_encoder saja untuk prediksi."""
        n = len(data)
        if n < self.seq_len:
            return np.zeros((0, self.seq_len, self.num_features))

        X_enc = []
        for i in range(n - self.seq_len + 1):
            X_enc.append(data[i : i + self.seq_len])
        return np.array(X_enc)

    @property
    def input_seq_len(self): return self.seq_len # panjang urutan input

    @property
    def target_seq_len(self): return self.pred_len # panjang urutan prediksi

# =============================================================================
# SECTION 4: DEEP LEARNING ARCHITECTURE (THE BRAIN)
# =============================================================================
# class untuk membangun arsitektur neural network LSTM dengan attention dan probabilistic output 
class DeepProbabilisticArchitecture:
    """
    Arsitektur Neural Network V6.0 Titan.
    Menggabungkan:
    - Bidirectional LSTM (Memahami konteks masa lalu & masa depan).
    - Bahdanau Attention (Fokus pada momen penting).
    - Probabilistic Output Layer (Prediksi Mean & Variance).
    """
    def __init__(self, config: LstmPipelineConfig):
        self.cfg = config
    # fungsi membangun model LSTM dengan attention dan output probabilistik 
    def build_model(self, num_features: int, params: Dict[str, Any] = None) -> Model:
        """
        Membangun Arsitektur Sequence-to-Sequence (Seq2Seq) dengan:
        1. Bi-LSTM Encoder
        2. LSTM Decoder (dengan Teacher Forcing inputs)
        3. Attention Mechanism (Bahdanau/Luong style)
        4. Probabilistic Output (Gaussian Layer: Mu & Sigma)
        """
        import tensorflow as tf # untuk operasi tensor
        from keras.layers import (Input, Dense, LSTM, Bidirectional, Conv1D, 
                                             BatchNormalization, Concatenate, Dropout, 
                                             TimeDistributed, Attention, Lambda) # untuk lapisan neural network
        from keras.models import Model # untuk definisi model
        from keras.optimizers import Adam # untuk optimizer
        import keras.backend as K # untuk fungsi backend Keras

        # Config Setup
        hp = params if params else {}
        input_len = self.cfg.input_seq_len
        target_len = self.cfg.target_seq_len
        
        # Hyperparameters (Prioritas: params > self.cfg > default)
        latent_dim = hp.get('latent_dim', getattr(self.cfg, 'latent_dim', 64))
        dropout = hp.get('dropout_rate', getattr(self.cfg, 'dropout_rate', 0.2))
        lr = hp.get('learning_rate', getattr(self.cfg, 'learning_rate', 0.001))
        
        # -----------------
        # 1. ENCODER BLOCK
        # -----------------
        encoder_inputs = Input(shape=(input_len, num_features), name='encoder_input') # input encoder
        
        # Feature Extraction (1D Conv) - Optional, bagus untuk menangkap pola lokal/noise
        x = Conv1D(filters=latent_dim, kernel_size=3, padding='same', activation='relu')(encoder_inputs)
        x = BatchNormalization()(x) #Rumus 3.28
        x = Dropout(dropout)(x)
        
        # Deep Bi-LSTM Encoder
        # Layer ini mengembalikan sequence untuk input attention
        # State h dan c digabungkan (Forward + Backward) untuk inisialisasi Decoder
        encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True, dropout=dropout), name='encoder_bi_lstm')
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(x)
        
        # Merge States: Karena Bidirectional, dimensi state menjadi 2x latent_dim
        state_h = Concatenate()([forward_h, backward_h]) #Rumus 3.29
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c] #Rumus 3.30, 3.31, 3.32

        # ---------------------------------------------------------
        # 2. DECODER BLOCK (Teacher Forcing Architecture)
        # ---------------------------------------------------------
        # Decoder input shape: (Prediction Length, Features)
        decoder_inputs = Input(shape=(target_len, num_features), name='decoder_input')
        
        # Decoder units harus match dengan encoder state size (latent_dim * 2)
        decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, dropout=dropout, name='decoder_lstm')
        
        # Output decoder mengabaikan state internalnya sendiri, melainkan diproses oleh Attention
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        # ---------------------------------------------------------
        # 3. ATTENTION MECHANISM
        # ---------------------------------------------------------
        # Attention layer menghubungkan:
        # Query = decoder_outputs (apa yang sedang kita prediksi sekarang)
        # Value = encoder_outputs (seluruh konteks masa lalu)
        attn_layer = Attention(name='attention_layer')
        context_vector = attn_layer([decoder_outputs, encoder_outputs])
        
        # Gabungkan Context Vector (dari masa lalu) + Output Decoder (prediksi saat ini)
        decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

        # ---------------------------------------------------------
        # 4. PROBABILISTIC HEAD (Aleatoric Uncertainty)
        # ---------------------------------------------------------
        # Layer Dense Intermediate
        x_out = TimeDistributed(Dense(64, activation='relu', kernel_initializer='he_normal'))(decoder_combined_context)
        x_out = Dropout(dropout)(x_out)
        
        # Head A: Prediksi Nilai Tengah (Mean / Mu)
        # HANYA INI YANG KITA PERTAHANKAN
        mu = TimeDistributed(Dense(1, activation='linear'), name='mu')(x_out)
        
        # HAPUS Head B, log_sigma_sq, Lambda Layer sigma, dan Concatenate
        # Output model HANYA mu
        output = mu
        
        # ---------------------------------------------------------
        # 5. COMPILATION
        # ---------------------------------------------------------
        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
        
        # Optimizer dengan gradient clipping untuk stabilitas
        optimizer = Adam(learning_rate=lr, clipnorm=1.0)
        
        try:
            # [FIX KRITIS]: Ganti Loss menjadi MAE/MSE standar
            model.compile(
                optimizer=optimizer,
                loss='mae', # Menggunakan Mean Absolute Error
                metrics=['mae', 'mse'] # Metrik standar
            )
            logger.info(f"Model LSTM berhasil dikompilasi (Standard MAE). Input: {input_len}x{num_features}")
        except Exception as e:
            logger.critical(f"FATAL: Gagal kompilasi model LSTM. Cek arsitektur. Error: {e}")
            raise RuntimeError(f"LSTM Compilation Failed: {e}")
            
        return model
# class untuk optimasi hyperparameter sederhana
class BayesianLikeOptimizer:
    """
    Sistem pencarian hyperparameter sederhana.
    Mencoba berbagai kombinasi konfigurasi untuk menemukan model terbaik.
    """
    def __init__(self, factory): #Rumus 3.33
        self.factory = factory
        self.space = {
            'latent_dim': [64, 128],
            'learning_rate': [5e-4, 1e-4, 5e-5], 
            'dropout_rate': [0.1, 0.2]
        }
    # fungsi pencarian hyperparameter
    def search(self, X_enc, X_dec, Y, trials=3):
        if len(X_enc) < 50: 
            return {'latent_dim': 64, 'learning_rate': 1e-3, 'dropout_rate': 0.2}
            
        best_loss = float('inf')
        best_params = {}
        
        logger.info(f"    [Tuner] Menjalankan {trials} trial optimasi...")
        # Loop pencarian hyperparameter
        for i in range(trials):
            params = {k: random.choice(v) for k, v in self.space.items()}
            K.clear_session() # Bersihkan memori GPU
            try:
                model = self.factory.build_model(X_enc.shape[-1], params)
                # Training singkat untuk evaluasi
                h = model.fit([X_enc, X_dec], Y, epochs=3, batch_size=32, verbose=0, validation_split=0.2)
                val_loss = h.history['val_loss'][-1]
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
            except: continue
        
        logger.info(f"    [Tuner] Parameter Terbaik: {best_params} (Loss: {best_loss:.4f})")
        return best_params

# =============================================================================
# SECTION 5: ARTIFACT & VISUALIZATION MANAGEMENT
# =============================================================================
# class untuk menyimpan dan memuat model, scaler, dan metadata
class ArtifactVault:
    """Menyimpan dan memuat model, scaler, dan metadata dengan aman."""
    def __init__(self, model_dir, visual_dir):
        self.model_dir = Path(model_dir)
        self.visual_dir = Path(visual_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)
    # fungsi menyimpan state cluster
    def save_cluster_state(self, cid, model, scaler, meta):
        try:
            model.save(self.model_dir / f"lstm_model_c{cid}.keras") # Simpan model
            dump(scaler, self.model_dir / f"scaler_c{cid}.joblib")  # Simpan scaler
            with open(self.model_dir / f"meta_c{cid}.json", 'w') as f: 
                json.dump(meta, f, indent=4)
        except Exception as e:
            logger.error(f"Save failed c{cid}: {e}")
    # fungsi memuat state cluster
    def load_cluster_state(self, cid):
        m_path = self.model_dir / f"lstm_model_c{cid}.keras" # path model
        s_path = self.model_dir / f"scaler_c{cid}.joblib" # path scaler
        if not m_path.exists(): # jika model tidak ada
            return None, None # kembalikan None

        try:
            import absl.logging # untuk menonaktifkan logging TF yang berlebihan
            absl.logging.set_verbosity(absl.logging.ERROR) # hanya error serius yang ditampilkan 

            # Define custom objects for loading model
            cust = {
                'gaussian_nll': MathKernel.gaussian_nll,
                'uncertainty_metric': MathKernel.uncertainty_metric,
                'mean_absolute_error_mu': MathKernel.mean_absolute_error_mu
            }

            # Safe loading: compile=False (jika versi TF lama/baru, opsi safe_mode mungkin tidak ada)
            model = load_model(m_path, custom_objects=cust, compile=False)
            scaler = load(s_path)
            
            logger.info(f"[Vault] Model Cluster {cid} berhasil dimuat.")
            return model, scaler
        except Exception as e:
            logger.error(f"[Vault] GAGAL memuat model c{cid}: {e}. File mungkin corrupt atau TF version mismatch.")
            return None, None
    # fungsi mendaftar cluster yang ada 
    def list_clusters(self):
        import re
        files = list(self.model_dir.glob("lstm_model_c*.keras"))
        clusters = []
        for f in files:
            m = re.search(r'c(\d+)\.keras$', f.name)
            if m: clusters.append(int(m.group(1)))
        return sorted(list(set(clusters)))
    # fungsi untuk alias kompatibilitas lama 
    def load_all(self, cid): return self.load_cluster_state(cid)
# class untuk visualisasi hasil prediksi dan pelatihan
class AdvancedVisualizer:
    """Generator grafik canggih untuk analisis probabilitas."""
    def __init__(self, output_dir):
        self.out = output_dir
    # fungsi plot forecast dengan interval ketidakpastian
    def plot_probabilistic_forecast(self, actual, pred_mu, pred_sigma, cid, suffix=""):
        try:
            plt.figure(figsize=(12, 6))
            x = np.arange(len(actual))
            plt.plot(x, actual, 'k-', label='Actual', alpha=0.7, linewidth=1.5)
            plt.plot(x, pred_mu, 'r--', label='Predicted Mean', linewidth=1.5)
            
            # Plot Uncertainty Interval (95% Confidence)
            lower = pred_mu - 1.96 * pred_sigma # 95% CI Lower Bound
            upper = pred_mu + 1.96 * pred_sigma # 95% CI Upper Bound
            plt.fill_between(x, lower, upper, color='red', alpha=0.2, label='Uncertainty (95% CI)') # Area ketidakpastian
            
            plt.title(f"Probabilistic Forecast - Cluster {cid} {suffix}", fontsize=14)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out / f"pred_vs_actual_c{cid}.png", dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Gagal plot forecast: {e}")
    # fungsi plot kurva loss selama pelatihan
    def plot_loss_curves(self, history: Dict, cid: int):
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(history['loss'], label='Train NLL')
            plt.plot(history['val_loss'], label='Val NLL')
            plt.title(f"Learning Curve (NLL) - Cluster {cid}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.out / f"loss_c{cid}.png", dpi=300)
            plt.close()
        except Exception: pass
    # fungsi plot distribusi residual error 
    def plot_residuals(self, errors: np.ndarray, cid: int):
        try:
            plt.figure(figsize=(8, 5))
            sns.histplot(errors, kde=True, color='purple')
            plt.title(f"Residual Distribution - Cluster {cid}")
            plt.xlabel("Error")
            plt.savefig(self.out / f"error_dist_c{cid}.png", dpi=300)
            plt.close()
        except Exception: pass
        
    # Aliases for compatibility with older code calls
    def plot_prediction_comparison(self, a, p, c): self.plot_probabilistic_forecast(a, p, np.zeros_like(p), c)
    def plot_error_distribution(self, e, c): self.plot_residuals(e, c)
    def plot_training_history(self, h, c): self.plot_loss_curves(h, c)

# =============================================================================
# SECTION 6: REALTIME BUFFER & DRIFT MONITOR
# =============================================================================
# class untuk memantau pergeseran data (data drift)
class DriftMonitor:
    """
    Memantau pergeseran data (Data Drift).
    Jika data realtime terlalu berbeda dari statistik data training,
    sistem akan memberikan peringatan.
    """
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.baseline_stats = {}
    # fungsi memperbarui baseline statistik
    def update_baseline(self, df_train: pd.DataFrame, features: List[str]):
        for f in features:
            if f in df_train:
                self.baseline_stats[f] = {
                    'mean': df_train[f].mean(),
                    'std': df_train[f].std()
                }

    def check_drift(self, df_new: pd.DataFrame) -> bool: # fungsi cek drift
        """Cek apakah data baru menyimpang jauh (Z-Score check)."""
        drift_detected = False # inisialisasi deteksi drift
        for f, stats in self.baseline_stats.items(): # iterasi fitur
            if f in df_new: # jika fitur ada di data baru
                val = df_new[f].mean() # ambil mean data baru
                z = abs(val - stats['mean']) / (stats['std'] + 1e-9) # hitung Z-Score
                if z > self.threshold: # jika Z-Score melebihi threshold
                    logger.warning(f"Data Drift Detected on {f} (Z={z:.2f})") # log peringatan
                    drift_detected = True # set deteksi drift
        return drift_detected # kembalikan hasil deteksi
# class untuk buffer memori jangka pendek (sliding window)
class InferenceBuffer:
    """
    Buffer Memori Jangka Pendek (Sliding Window).
    Menampung data history untuk memberikan konteks pada prediksi realtime.
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer_df = pd.DataFrame()
    # fungsi memperbarui buffer dengan data baru
    def update(self, df_new):
        if df_new.empty: return
        self.buffer_df = pd.concat([self.buffer_df, df_new], ignore_index=True)
        self.buffer_df = self.buffer_df.sort_values('Acquired_Date')
        self.buffer_df = self.buffer_df.drop_duplicates(subset=['Acquired_Date', 'Nama'], keep='last')
        
        # Keep buffer size limited (e.g., 3x window size)
        limit = self.window_size * 3
        if len(self.buffer_df) > limit:
            self.buffer_df = self.buffer_df.iloc[-limit:]
     # fungsi mendapatkan salinan buffer saat ini       
    def get_context(self):
        return self.buffer_df.copy()
# class untuk memproses data (feature engineering dan clustering)
class DataProcessor:
    """Wrapper untuk Feature Engineering di dalam LSTM."""
    def __init__(self, config):
        self.cfg = config
        # Feature Engineer diinisialisasi dengan Config Global
        self.fe = FeatureEngineer(CONFIG.FEATURE_ENGINEERING, CONFIG.ACO_ENGINE)
        self.guard = DataGuard(['Acquired_Date', 'Magnitudo'])
        self.clusterer = GeoClusterer(config.clustering_eps, config.clustering_min_samples)
    # fungsi persiapan data standar 
    def prepare(self, df):
        """Pipeline preprocessing standar."""
        df = self.guard.sanitize_temporal(df, 'Acquired_Date')
        df = self.guard.sanitize_numeric(df, ['Magnitudo'])
        df = self.fe.basic_cleanup(df)
        df = self.fe.add_spatio_temporal_features(df)
        df = self.fe.add_lag_and_rolling(df)
        
        if 'cluster_id' not in df.columns:
            df['cluster_id'] = self.clusterer.fit_predict(df)
        return df

# =============================================================================
# SECTION 7: MAIN ENGINE FACADE (THE INTERFACE)
# =============================================================================
# class utama untuk menghubungkan semua komponen LSTM menjadi satu kesatuan cerdas 
class LstmEngine:
    """
    Engine Utama LSTM V6.0 TITAN.
    Menghubungkan semua komponen di atas menjadi satu kesatuan cerdas.
    """
    def __init__(self, config):
        self.cfg = config
        self.vault = ArtifactVault(self.cfg.model_dir, self.cfg.visuals_dir) # manajer artefak
        self.processor = DataProcessor(self.cfg) # prosesor data
        self.architect = DeepProbabilisticArchitecture(self.cfg) # arsitektur neural network
        self.tuner = BayesianLikeOptimizer(self.architect) # optimizer hyperparameter
        self.viz_manager = AdvancedVisualizer(Path(self.cfg.visuals_dir)) # manajer visualisasi
        self.buffer = InferenceBuffer(self.cfg.input_seq_len) # buffer inferensi
        self.drift_mon = DriftMonitor() # monitor drift data
        
        # Cache in-memory untuk performa realtime
        self.models_cache = {}
        # Bersihkan folder logs lama
        if os.path.exists("logs"): shutil.rmtree("logs", ignore_errors=True)

    # Compatibility Props for other engines (Tidak ada perubahan)
    @property
    def manager(self): return self.vault 
    @property
    def trainer(self): return self
    @property
    def viz(self): return self.viz_manager
    # fungsi memuat data ke buffer
    def load_buffer(self, df_history):
        """Memuat data training ke buffer (inisialisasi untuk live stream)."""
        if df_history is not None and not df_history.empty:
            logger.info(f"Loading {len(df_history)} rows to buffer.")
            self.buffer.update(df_history)
            
            # Init drift baseline from history
            df_proc = self.processor.prepare(df_history)
            feats = [c for c in df_proc.columns if c in self.cfg.features]
            self.drift_mon.update_baseline(df_proc, feats)

    def integrate_aco_prediction(self, pred: Dict[str, Any], cid: Optional[int] = None, attach_to: str = "nearest"):
        """
        Integrasi output ACO ke buffer LSTM.
        pred: dict mis. {'center_lat':.., 'center_lon':.., 'area_km2':.., 'confidence':..}
        attach_to: "nearest" | "append_row"
        """
        try:
            if not pred or ('center_lat' not in pred or 'center_lon' not in pred):
                logger.warning("[LSTM] integrate_aco_prediction: pred kosong atau tidak punya center coords.")
                return None

            buf = self.buffer.get_context()
            if buf is None or buf.empty:
                logger.warning("[LSTM] Buffer kosong, mencoba append_row.")
                if attach_to == "append_row":
                    row = {
                        'Acquired_Date': pd.Timestamp.now(),
                        'EQ_Lintang': pred.get('center_lat'),
                        'EQ_Bujur': pred.get('center_lon'),
                        'Nama': 'ACO_PRED',
                        'aco_center_lat': pred.get('center_lat'),
                        'aco_center_lon': pred.get('center_lon'),
                        'aco_area_km2': pred.get('area_km2'),
                        'aco_confidence': pred.get('confidence', 1.0)
                    }
                    self.buffer.update(pd.DataFrame([row]))
                    return True
                return None

            # Cari kandidat (cluster filter bila cid disediakan)
            candidates = buf if cid is None else (buf[buf['cluster_id'] == cid] if 'cluster_id' in buf.columns else buf)
            coords = candidates[['EQ_Lintang', 'EQ_Bujur']].dropna()
            if coords.empty:
                if attach_to == "append_row":
                    row = {'Acquired_Date': pd.Timestamp.now(), 'EQ_Lintang': pred.get('center_lat'), 'EQ_Bujur': pred.get('center_lon'), 'Nama': 'ACO_PRED'}
                    row.update({
                        'aco_center_lat': pred.get('center_lat'),
                        'aco_center_lon': pred.get('center_lon'),
                        'aco_area_km2': pred.get('area_km2'),
                        'aco_confidence': pred.get('confidence', 1.0)
                    })
                    self.buffer.update(pd.DataFrame([row]))
                    return True
                return None

            lat_p = float(pred.get('center_lat'))
            lon_p = float(pred.get('center_lon'))
            lat_arr = coords['EQ_Lintang'].astype(float).values
            lon_arr = coords['EQ_Bujur'].astype(float).values

            # gunakan GeoMathCore.haversine jika tersedia, fallback simple euclidean approx jika tidak
            try:
                dists = np.array([GeoMathCore.haversine(lat_p, lon_p, la, lo) for la, lo in zip(lat_arr, lon_arr)])
            except Exception:
                dists = np.sqrt((lat_arr - lat_p)**2 + (lon_arr - lon_p)**2)

            nearest_idx_local = coords.index[np.argmin(dists)]

            self.buffer.buffer_df.loc[nearest_idx_local, 'aco_center_lat'] = lat_p
            self.buffer.buffer_df.loc[nearest_idx_local, 'aco_center_lon'] = lon_p
            self.buffer.buffer_df.loc[nearest_idx_local, 'aco_area_km2'] = pred.get('area_km2')
            self.buffer.buffer_df.loc[nearest_idx_local, 'aco_confidence'] = pred.get('confidence', 1.0)

            logger.info(f"[LSTM] ACO pred integrated to buffer index {nearest_idx_local} (cid={cid})")
            return nearest_idx_local

        except Exception as e:
            logger.error(f"[LSTM] integrate_aco_prediction failed: {e}")
            return None


    # fungsi integrasi prediksi GA ke buffer
    def integrate_ga_prediction(self, pred: Dict[str, Any], cid: Optional[int] = None, attach_to: str = "nearest"):
        """
        Integrasi output GA (pred dict) ke buffer LSTM.
        - pred: dict seperti {'pred_lat', 'pred_lon', 'bearing_degree', 'distance_km', 'confidence', ...}
        - cid: jika known cluster_id, gunakan mapping langsung; jika None, akan dicari row terdekat di buffer
        - attach_to: "nearest" | "append_row"
        Efek: menambah kolom GA ke baris yang relevan di self.buffer.buffer_df
        """
        try: # validasi input pred 
            if not pred or ('pred_lat' not in pred or 'pred_lon' not in pred): # jika pred kosong atau tidak ada lat/lon
                logger.warning("[LSTM] integrate_ga_prediction: pred kosong atau tidak punya lat/lon.") # log peringatan
                return None # kembalikan None
            # ambil salinan buffer saat ini
            buf = self.buffer.get_context() # ambil salinan buffer
            if buf is None or buf.empty: # jika buffer kosong
                logger.warning("[LSTM] Buffer kosong, tidak ada tempat integrasi GA; opsi append_row dipertimbangkan.")
                if attach_to == "append_row":
                    row = {
                        'Acquired_Date': pd.Timestamp.now(),
                        'EQ_Lintang': pred.get('pred_lat'),
                        'EQ_Bujur': pred.get('pred_lon'),
                        'Nama': 'GA_PRED',
                    } # buat row baru
                    row.update({
                        'ga_pred_lat': pred.get('pred_lat'),
                        'ga_pred_lon': pred.get('pred_lon'),
                        'ga_bearing': pred.get('bearing_degree'),
                        'ga_distance_km': pred.get('distance_km'),
                        'ga_confidence': pred.get('confidence'),
                    }) # update row dengan data GA
                    self.buffer.update(pd.DataFrame([row]))
                    return True
                return None

            # jika cluster id diberikan -> filter buffer per cluster
            if cid is not None and 'cluster_id' in buf.columns:
                sub = buf[buf['cluster_id'] == cid]
                if sub.empty: 
                    candidates = buf
                else:
                    candidates = sub
            else:
                candidates = buf

            # hitung index terdekat (haversine) ke pred point
            lat_p = float(pred.get('pred_lat')) 
            lon_p = float(pred.get('pred_lon'))

            # vectorized haversine
            coords = candidates[['EQ_Lintang', 'EQ_Bujur']].dropna() # ambil koordinat valid 
            if coords.empty: # jika tidak ada koordinat valid
                # fallback append
                if attach_to == "append_row": # jika opsi append_row
                    row = {'Acquired_Date': pd.Timestamp.now(), 'EQ_Lintang': lat_p, 'EQ_Bujur': lon_p, 'Nama': 'GA_PRED'} # buat row baru
                    row.update({
                        'ga_pred_lat': lat_p,
                        'ga_pred_lon': lon_p,
                        'ga_bearing': pred.get('bearing_degree'),
                        'ga_distance_km': pred.get('distance_km'),
                        'ga_confidence': pred.get('confidence'),
                    }) # update row dengan data GA
                    self.buffer.update(pd.DataFrame([row]))
                    return True
                return None

            # fungsi untuk hitung haversine (tidak dipakai, hanya referensi)
            def _h(lat1, lon1, lat2_arr, lon2_arr):
                return np.array([GeoClusterer(0,1).model.metric if False else
                                 GeoMathCore.haversine(lat1, lon1, rlat, rlon)
                                 for rlat, rlon in zip(lat2_arr, lon2_arr)])

            # faster vector compute
            lat_arr = coords['EQ_Lintang'].astype(float).values # ambil array lat
            lon_arr = coords['EQ_Bujur'].astype(float).values # ambil array lon
            # compute distances vectorized using GeoMathCore.haversine in loop (numpy vectorization with listcomp)
            dists = np.array([GeoMathCore.haversine(lat_p, lon_p, la, lo) for la, lo in zip(lat_arr, lon_arr)]) # hitung jarak
            nearest_idx_local = coords.index[np.argmin(dists)] # ambil index terdekat

            # update buffer row with GA fields
            self.buffer.buffer_df.loc[nearest_idx_local, 'ga_pred_lat'] = lat_p # update lat
            self.buffer.buffer_df.loc[nearest_idx_local, 'ga_pred_lon'] = lon_p # update lon
            self.buffer.buffer_df.loc[nearest_idx_local, 'ga_bearing'] = pred.get('bearing_degree') # update bearing
            self.buffer.buffer_df.loc[nearest_idx_local, 'ga_distance_km'] = pred.get('distance_km') # update distance
            self.buffer.buffer_df.loc[nearest_idx_local, 'ga_confidence'] = pred.get('confidence') # update confidence

            logger.info(f"[LSTM] GA pred integrated to buffer index {nearest_idx_local} (cid={cid})")
            return nearest_idx_local

        except Exception as e:
            logger.error(f"[LSTM] integrate_ga_prediction failed: {e}")
            return None

    def load_ga_json_and_integrate(self, ga_json_path: str, cid: Optional[int] = None): # fungsi muat file JSON GA dan integrasi ke buffer 
        """
        Load GA vector JSON file and integrate into buffer.
        Returns True/False
        """
        try:
            if not os.path.exists(ga_json_path):
                logger.warning(f"[LSTM] GA json not found: {ga_json_path}")
                return False
            with open(ga_json_path, 'r') as f:
                pred = json.load(f)
            if isinstance(pred, dict) and 'pred_lat' in pred:
                return bool(self.integrate_ga_prediction(pred, cid=cid))
            logger.warning("[LSTM] GA JSON doesn't contain 'pred_lat'/'pred_lon'")
            return False
        except Exception as e:
            logger.error(f"[LSTM] load_ga_json_and_integrate failed: {e}")
            return False
    # Tambahkan method ini di bawah integrate_ga_prediction
    def integrate_cnn_prediction(self, pred: Dict[str, Any], attach_to: str = "nearest"):
        try:
            if not pred: return None
            buf = self.buffer.get_context()
            if buf is None or buf.empty:
                return None

            # Jika payload CNN berisi coords, gunakan nearest; else gunakan latest
            if 'pred_lat' in pred and 'pred_lon' in pred:
                coords = buf[['EQ_Lintang', 'EQ_Bujur']].dropna()
                if not coords.empty:
                    lat_p = float(pred.get('pred_lat'))
                    lon_p = float(pred.get('pred_lon'))
                    lat_arr = coords['EQ_Lintang'].astype(float).values
                    lon_arr = coords['EQ_Bujur'].astype(float).values
                    try:
                        dists = np.array([GeoMathCore.haversine(lat_p, lon_p, la, lo) for la, lo in zip(lat_arr, lon_arr)])
                    except Exception:
                        dists = np.sqrt((lat_arr - lat_p)**2 + (lon_arr - lon_p)**2)
                    nearest_idx_local = coords.index[np.argmin(dists)]
                    idx_to_update = nearest_idx_local
                else:
                    idx_to_update = buf.index[-1]
            else:
                idx_to_update = buf.index[-1]

            self.buffer.buffer_df.loc[idx_to_update, 'cnn_bearing'] = pred.get('bearing_degree', pred.get('predicted_bearing'))
            self.buffer.buffer_df.loc[idx_to_update, 'cnn_distance'] = pred.get('distance_km', pred.get('predicted_distance'))
            self.buffer.buffer_df.loc[idx_to_update, 'cnn_confidence'] = pred.get('confidence', 0.0)

            logger.info(f"[LSTM] CNN pred integrated to buffer index {idx_to_update}")
            return idx_to_update

        except Exception as e:
            logger.error(f"[LSTM] integrate_cnn_prediction failed: {e}")
            return None

    # fungsi mendapatkan data buffer 
    def get_buffer(self) -> pd.DataFrame:
        """Mengembalikan data buffer historis dari InferenceBuffer."""
        return self.buffer.get_context()
    # fungsi memperbarui buffer dengan event baru
    def update_buffer(self, df_new_events: pd.DataFrame):
        """Memperbarui buffer dengan event yang baru diprediksi."""
        self.buffer.update(df_new_events)
    # fungsi pelatihan per-cluster (dummy untuk kompatibilitas)
    def train_cluster(self, cid, data, scaler): pass # Dummy for compat

    @execution_telemetry
    def train(self, df_train): # fungsi pelatihan model LSTM per-cluster Rumus 3.26 dan 3.27
        """
        Melatih model LSTM per-cluster dengan strategi cleaning & scaling yang robust.
        """
        if df_train is None or df_train.empty: 
            return False
        
        logger.info("=== LSTM V6.0 TITAN TRAINING START ===")
        
        # 1. Prepare data
        df_proc = self.processor.prepare(df_train)
        
        # Ambil list cluster valid (kecuali noise -1)
        clusters = sorted([c for c in df_proc['cluster_id'].unique() if c != -1])
        success = 0
        
        for cid in clusters:
            logger.info(f"Processing Cluster {cid}...")
            df_c = df_proc[df_proc['cluster_id'] == cid].sort_values('Acquired_Date')
            
            # Filter fitur
            ga_cols = [c for c in df_c.columns if c.startswith('ga_') or c in
                       ('ga_pred_lat', 'ga_pred_lon', 'ga_bearing', 'ga_distance_km', 'ga_confidence')]
            feats = [c for c in df_c.columns if (c in self.cfg.features or c == self.cfg.target_feature)]
            # union with ga cols
            feats = list(set(feats + ga_cols))
            
            # --- 1. ROBUST DATA CLEANING ---
            data_to_process = df_c[feats].copy()
            
            # [Step A]: Sanitasi Nilai Ekstrim
            data_to_process = data_to_process.replace([np.inf, -np.inf], np.nan)
            
            # [Step B]: Imputasi Prioritas Median
            # Cara ini jauh lebih stabil dan cepat daripada KNNImputer di dalam loop
            median_vals = data_to_process.median(numeric_only=True)
            data_to_process = data_to_process.fillna(median_vals)
            
            # Final fallback: Isi 0.0 jika masih ada NaN (misal kolom kosong total)
            data_to_process = data_to_process.fillna(0.0)

            # --- 2. SCALING ---
            std_sum = data_to_process.std(numeric_only=True).sum()
            if std_sum < 1e-9:
                logger.warning(f"Cluster {cid} diabaikan: Data Konstan/Flat (Total StdDev={std_sum:.2e}).")
                continue # Skip cluster ini, jangan paksa training

            # Inisialisasi Scaler
            # Note: RobustScaler lebih disarankan untuk data seismik dibanding StandardScaler
            # karena gempa besar adalah outlier alami. RobustScaler menggunakan median & IQR.
            scaler = RobustScaler()

            try:
                # [Primary Strategy]: RobustScaler
                # Ideal untuk data seismik karena menggunakan Median & IQR (kebal terhadap gempa besar/outlier)
                data_mtx = scaler.fit_transform(data_to_process)
                
            except Exception as e_robust:
                # [Fallback Strategy]: StandardScaler
                # Digunakan jika RobustScaler gagal (misalnya karena Interquartile Range = 0)
                logger.warning(f"RobustScaler issue at Cluster {cid} ({e_robust}). Fallback ke StandardScaler.")
                
                try:
                    scaler = StandardScaler()
                    data_mtx = scaler.fit_transform(data_to_process)
                except Exception as e_std:
                    logger.error(f"Scaling FATAL failure c{cid}: {e_std}. Skip cluster.")
                    continue

            # [FIX KRITIS]: Post-Scaling Safety Check
            # Cek apakah hasil scaling mengandung NaN atau Infinity
            if not np.isfinite(data_mtx).all():
                logger.error(f"Scaling result contains NaN/Inf at cluster {cid}. Skip cluster.")
                continue

            # --- 3. TENSOR CREATION ---
            tfactory = TensorFactory(feats, self.cfg.target_feature, self.cfg.input_seq_len, self.cfg.target_seq_len)
            
            try:
                X_enc, X_dec, Y = tfactory.construct_training_tensors(data_mtx)
            except Exception as e:
                logger.warning(f"Tensor construct error c{cid}: {e}. Skip.")
                continue
            
            # Skip jika sampel terlalu sedikit
            if len(X_enc) < 10: 
                logger.warning(f"Sample c{cid} terlalu sedikit ({len(X_enc)}). Skip training.")
                continue
            
            # --- 4. MODEL SEARCH & TRAINING ---
            best_p = self.tuner.search(X_enc, X_dec, Y)
            model = self.architect.build_model(len(feats), best_p)
            
            log_path = Path("logs") / f"c{cid}"
            
            cbs = [
                EarlyStopping(patience=self.cfg.early_stopping_patience, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
                TensorBoard(log_dir=str(log_path))
            ]
            
            try:
                h = model.fit(
                    [X_enc, X_dec], Y, 
                    epochs=self.cfg.epochs, 
                    batch_size=self.cfg.batch_size, 
                    validation_split=self.cfg.validation_split, 
                    callbacks=cbs, 
                    verbose=1
                )
                
                # Simpan State
                meta = {"features": feats, "best_params": best_p, "trained_at": str(datetime.now())}
                self.vault.save_cluster_state(cid, model, scaler, meta)
                
                # Plot Curve
                self.viz_manager.plot_loss_curves(h.history, cid)
                
                success += 1
            except Exception as e:
                logger.error(f"Training Failed Cluster {cid}: {e}")

        return success > 0

    @execution_telemetry
    def predict_on_static(self, df_test): #Rumus 3.34
        # [FIX] Pastikan kita punya data input
        if df_test is None or df_test.empty: 
            return df_test, pd.DataFrame() 

        # 1. SIMPAN DATA ASLI (RAW) & URUTKAN WAKTU
        # Ini penting agar kolom ACO/GA/CNN bawaan tidak hilang
        df_raw = df_test.copy()
        if 'Acquired_Date' in df_raw.columns:
            df_raw['Acquired_Date'] = pd.to_datetime(df_raw['Acquired_Date'], errors='coerce')
            df_raw = df_raw.sort_values('Acquired_Date').reset_index(drop=True)
        
        # 2. PROSES DATA UNTUK MODEL (Cleaning & Scaling)
        # df_proc ini hanya berisi angka-angka untuk input LSTM, kolom lain dibuang disini
        df_proc = self.processor.prepare(df_raw) 
        
        # Kita butuh df_out sebagai wadah hasil prediksi, tapi basisnya tetap data asli
        # Supaya aman, kita clone df_raw sebagai base output
        df_out = df_raw.copy()

        # Inisialisasi kolom hasil jika belum ada
        cols_to_init = ['lstm_prediction', 'prediction_sigma', 'prediction_error', 'anomaly_score']
        for c in cols_to_init:
            df_out[c] = np.nan
        df_out['anomaly_score'] = 0.0

        # --- MULAI PROSES PREDIKSI (Sama seperti sebelumnya) ---
        anomalies = [] 
        if 'cluster_id' not in df_proc.columns: df_proc['cluster_id'] = -1 

        # Kita loop berdasarkan cluster, tapi update-nya ke df_out (yang berisi data asli)
        for cid in self.vault.list_clusters(): 
            # Ambil data yang sudah diproses untuk masuk ke model
            mask_proc = df_proc['cluster_id'] == cid 
            if not mask_proc.any(): continue
            
            df_c_proc = df_proc.loc[mask_proc].sort_values('Acquired_Date')

            # Load Model
            if cid in self.models_cache: 
                model, scaler = self.models_cache[cid] 
            else: 
                model, scaler = self.vault.load_cluster_state(cid) 
                if model: self.models_cache[cid] = (model, scaler) 

            if not model: continue 

            # Validasi Fitur
            feats = getattr(scaler, 'feature_names_in_', list(self.cfg.features)) 
            
            # Handling kolom GA/ACO yang mungkin jadi fitur tambahan
            extra_cols = [c for c in df_c_proc.columns if any(x in c for x in ['ga_', 'aco_'])]
            current_feats = list(feats)
            for ec in extra_cols:
                if ec not in current_feats: current_feats.append(ec)

            # Siapkan data untuk transform
            data_subset = df_c_proc.reindex(columns=current_feats).fillna(0)
            
            # Jika fitur tidak cocok dengan scaler, paksa sesuaikan
            if data_subset.shape[1] != scaler.n_features_in_:
                 valid_feats = getattr(scaler, 'feature_names_in_', self.cfg.features)
                 data_subset = df_c_proc.reindex(columns=valid_feats).fillna(0)

            try:
                data_mtx = scaler.transform(data_subset) 
            except Exception:
                continue

            # Buat Tensor
            tfactory = TensorFactory(list(data_subset.columns), self.cfg.target_feature, self.cfg.input_seq_len, self.cfg.target_seq_len) 
            X_enc = tfactory.construct_inference_tensor(data_mtx) 
            # =====================================================
            # [GA TIME-SERIES AWARE INPUT ADAPTER — SAFE MODE]
            # =====================================================
            try:
                expected_feats = model.input_shape[-1]  # contoh: 6
                current_feats = X_enc.shape[-1]

                if current_feats == expected_feats:
                    # AMAN → model memang dilatih dengan GA
                    ga_event_path = os.path.join("output", "ga_results", "ga_events.csv")

                    if os.path.exists(ga_event_path):
                        df_ga_evt = pd.read_csv(ga_event_path, parse_dates=["timestamp"])
                        df_ga_evt = df_ga_evt.sort_values("timestamp")

                        ga_feats = df_ga_evt[["ga_bearing_deg", "ga_distance_km"]].values
                        window = self.cfg.input_seq_len

                        if len(ga_feats) < window:
                            pad = np.zeros((window - len(ga_feats), 2))
                            ga_feats = np.vstack([pad, ga_feats])

                        ga_seq = ga_feats[-window:]
                        ga_seq = ga_seq.reshape(1, window, 2)
                        ga_seq = np.repeat(ga_seq, X_enc.shape[0], axis=0)

                        X_enc = np.concatenate([X_enc, ga_seq], axis=-1)

                        logger.info("[LSTM] GA time-series injected into encoder input")

                else:
                    # MODEL LAMA → JANGAN DIPAKSA
                    logger.warning(
                        f"[LSTM] GA NOT injected. Model expects {expected_feats} features, "
                        f"but GA would increase to {current_feats + 2}. Retrain required."
                    )

            except Exception as e:
                logger.warning(f"[LSTM] GA adapter skipped: {e}")

            if len(X_enc) == 0: continue 

            # Predict
            X_dec_dummy = np.zeros((len(X_enc), self.cfg.target_seq_len, X_enc.shape[2])) 
            preds = model.predict([X_enc, X_dec_dummy], verbose=0) 
            preds = np.squeeze(preds)
            if preds.ndim == 2: mu_seq = preds[:, 0]
            else: mu_seq = preds  

            # Inverse Scale Target
            from sklearn.preprocessing import MinMaxScaler 
            target_scaler = MinMaxScaler()
            target_vals = df_c_proc[[self.cfg.target_feature]].values
            if len(target_vals) > 0:
                target_scaler.fit(target_vals)
                res_mu = target_scaler.inverse_transform(mu_seq.reshape(-1, 1)).ravel()
            else:
                res_mu = mu_seq # Fallback

            # Mapping Index Hasil Prediksi ke DataFrame Asli
            # Kita gunakan index dari df_c_proc karena index-nya inherit dari df_raw (reset_index(drop=True) diatas)
            # Hati-hati: df_proc mungkin punya index berbeda kalau di filter
            # Strategi aman: Gunakan Acquired_Date untuk mapping balik
            
            start_idx = self.cfg.input_seq_len
            valid_dates = df_c_proc['Acquired_Date'].iloc[start_idx : start_idx + len(res_mu)]
            
            # Update ke df_out berdasarkan Tanggal yang cocok
            # Ini memastikan kita menempelkan hasil ke baris yang benar di data asli
            common_dates = valid_dates[valid_dates.isin(df_out['Acquired_Date'])]
            
            if len(common_dates) > 0:
                # Ambil nilai aktual untuk error
                actual = df_out.loc[df_out['Acquired_Date'].isin(common_dates), self.cfg.target_feature].values
                pred_vals = res_mu[:len(common_dates)]
                
                err = np.abs(pred_vals - actual)
                
                # Update DataFrame
                df_out.loc[df_out['Acquired_Date'].isin(common_dates), 'lstm_prediction'] = pred_vals
                df_out.loc[df_out['Acquired_Date'].isin(common_dates), 'prediction_error'] = err
                
                # Anomaly Score
                err_mean = np.mean(err)
                err_std = np.std(err)
                if err_std < 1e-6: z_score = np.zeros_like(err)
                else: z_score = (err - err_mean) / err_std
                
                df_out.loc[df_out['Acquired_Date'].isin(common_dates), 'anomaly_score'] = z_score

                # Collect Anomalies
                idx_anom = df_out.loc[df_out['Acquired_Date'].isin(common_dates)][z_score > 2.5].index #Rumus 3.35
                if len(idx_anom) > 0:
                    anomalies.append(df_out.loc[idx_anom])

        # 3. INTEGRASI DATA EKSTERNAL (UPDATE LAST ROW DARI JSON)
        # Ini menimpa baris terakhir dengan data realtime dari JSON (GA/ACO/CNN)
        # hanya jika file JSON ada.
        try:
            if not df_out.empty:
                last_idx = df_out.index[-1]
                
                # --- ACO ---
                aco_path = os.path.join("output", "aco_results", "aco_to_ga.json")
                if os.path.exists(aco_path):
                    with open(aco_path, 'r') as f: aco_data = json.load(f)
                    # Timpa nilai hanya jika JSON valid
                    if 'center_lat' in aco_data:
                        df_out.loc[last_idx, 'aco_center_lat'] = aco_data.get('center_lat')
                        df_out.loc[last_idx, 'aco_center_lon'] = aco_data.get('center_lon')
                        df_out.loc[last_idx, 'aco_area_km2'] = aco_data.get('area_km2')

                # --- GA ---
                ga_path = os.path.join("output", "ga_results", "ga_vector.json")
                if os.path.exists(ga_path):
                    with open(ga_path, 'r') as f: ga_data = json.load(f)
                    if 'bearing_degree' in ga_data:
                        df_out.loc[last_idx, 'ga_bearing'] = ga_data.get('bearing_degree')
                        df_out.loc[last_idx, 'ga_distance_km'] = ga_data.get('distance_km')

                # --- CNN ---
                cnn_path = os.path.join("output", "cnn_results", "cnn_predictions_latest.json")
                if os.path.exists(cnn_path):
                    with open(cnn_path, 'r') as f: cnn_data = json.load(f)
                    # Handle structure variation
                    val_bearing = cnn_data.get('next_event', {}).get('direction_deg', cnn_data.get('bearing_degree'))
                    val_dist = cnn_data.get('next_event', {}).get('distance_km', cnn_data.get('distance_km'))
                    
                    if val_bearing is not None:
                         # Kita simpan di kolom cnn_bearing (atau timpa ga_bearing jika diminta client)
                         # Sesuai request: "hasil GA sudut dan arah" -> biarkan kolom ga_
                         # Tapi "output CNN arah dan sudut" -> kita simpan juga
                         df_out.loc[last_idx, 'cnn_bearing'] = val_bearing
                         df_out.loc[last_idx, 'cnn_distance'] = val_dist

        except Exception as e:
            logger.warning(f"[LSTM] Integration error: {e}")

        # Gabungkan semua anomali yang ditemukan
        final_anoms = pd.concat(anomalies) if anomalies else pd.DataFrame() 

        # 4. SAVE (Panggil fungsi save yang baru)
        try: 
            self._save_lstm_records(df_out, final_anoms)
        except Exception as e:
            logger.warning(f"[LSTM] Failed to save LSTM records: {e}")

        return df_out, final_anoms

    # fungsi merekam event aktual ke buffer 
    def record_actual_events(self, df_actual: pd.DataFrame):
        """
        Merekam event aktual (ground truth) ke buffer & vault
        untuk pembelajaran lanjutan LSTM (integrasi CNN).
        """
        if df_actual is None or df_actual.empty: # jika data kosong
            logger.warning("[LSTM] record_actual_events: df_actual kosong") # log peringatan
            return False # kembalikan False

        try: # proses perekaman untuk event aktual 
            # 1️⃣ Update buffer realtime
            self.buffer.update(df_actual)

            # 2️⃣ (Opsional) Simpan ke vault sebagai arsip learning
            if hasattr(self.vault, 'append'):
                self.vault.append(df_actual)

            # 3️⃣ Persist state
            self.save_state()

            logger.info(f"[LSTM] Recorded {len(df_actual)} actual events")
            return True

        except Exception as e:
            logger.error(f"[LSTM] record_actual_events failed: {e}")
            return False

    # fungsi simpan state LSTM
    def save_state(self):
        """
        Simpan state LSTM (buffer, metadata).
        """
        try:
            # Simpan buffer ke pickle
            state_dir = Path(self.cfg.model_dir)
            state_dir.mkdir(parents=True, exist_ok=True)

            buffer_path = state_dir / "lstm_buffer.pkl"
            with open(buffer_path, "wb") as f:
                pickle.dump(self.buffer.get_context(), f)

            logger.info(f"[LSTM] State saved: {buffer_path}")

        except Exception as e:
            logger.warning(f"[LSTM] save_state failed: {e}")

    def extract_hidden_states(self, model, X_enc, X_dec=None): # Tambahkan =None
        """
        Mengambil hidden state dari encoder LSTM
        """
        try:
            # Handle jika X_dec tidak diberikan oleh pemanggil
            if X_dec is None:
                # Buat dummy X_dec sesuai shape yang diharapkan model (batch_size, target_len, features)
                # Kita ambil shape dari X_enc: (batch, seq_len, features)
                batch_size = X_enc.shape[0]
                n_features = X_enc.shape[2]
                target_len = self.cfg.target_seq_len
                X_dec = np.zeros((batch_size, target_len, n_features))
            # Ambil layer encoder
            enc_layer = model.get_layer('encoder_bi_lstm')
            enc_model = Model(
                inputs=model.input,
                outputs=enc_layer.output
            )

            outputs = enc_model.predict([X_enc, X_dec], verbose=0)
            return outputs

        except Exception as e:
            logger.warning(f"[LSTM] extract_hidden_states failed: {e}")
            return None

    # fungsi proses data live stream 
    def process_live_stream(self, df_new):
        if df_new.empty: return pd.DataFrame(), pd.DataFrame()
        
        # Check Drift
        self.drift_mon.check_drift(df_new)
        
        self.buffer.update(df_new)
        ctx = self.buffer.get_context()
        
        pred, anom = self.predict_on_static(ctx)
        
        # Filter results for new data only
        new_ts = df_new['Acquired_Date'].values
        final_pred = pred[pred['Acquired_Date'].isin(new_ts)]
        
        if not anom.empty and 'Acquired_Date' in anom.columns:
            final_anom = anom[anom['Acquired_Date'].isin(new_ts)]
        else:
            final_anom = pd.DataFrame()
        
        # Simpan record terbaru & anomalies ke CSV agar downstream (CNN / NB) bisa konsumsi
        try:
            self._save_lstm_records(ctx, final_anom)
        except Exception as e:
            logger.warning(f"[LSTM] Failed auto-save during live stream: {e}")

        return final_pred, final_anom

    # fungsi simpan record LSTM ke CSV 
    def _save_lstm_records(self, df_full: pd.DataFrame, anomalies: pd.DataFrame):
        """
        [MODIFIED V4] Menyimpan output untuk Client DAN output khusus untuk CNN Engine.
        """
        # Direktori output utama untuk hasil LSTM
        out_root = getattr(self.cfg, 'output_dir', 'output/lstm_results')
        os.makedirs(out_root, exist_ok=True)

        # 1. Siapkan DataFrame Utama
        df = df_full.copy()
        
        # Validasi Kolom Waktu
        if 'Acquired_Date' not in df.columns:
            logger.error("[LSTM] Acquired_Date tidak ditemukan.")
            return {}

        df['Acquired_Date'] = pd.to_datetime(df['Acquired_Date'], errors='coerce')
        df = df.dropna(subset=['Acquired_Date']).sort_values("Acquired_Date")
        
        # =====================================================================
        # BAGIAN 1: LOAD & MERGE DATA ACO (Zoning Data)
        # =====================================================================
        try:
            aco_source = None
            
            # Daftar lokasi file ACO
            aco_search_paths = [ #input LSTM
                r"output/aco_results/aco_zoning_data_for_lstm.xlsx",
                r"output/aco_results/aco_zoning_data_for_lstm.csv",
                os.path.join(out_root, "aco_zoning_data_for_lstm.xlsx"),
                "aco_zoning_data_for_lstm.xlsx"
            ]

            found_aco = None #Rumus 3.24 dan 3.25
            for p in aco_search_paths:
                norm_p = os.path.normpath(p) 
                if os.path.exists(norm_p):
                    found_aco = norm_p
                    break
            
            if found_aco:
                logger.info(f"[LSTM] Membaca data ACO dari: {found_aco}")
                if found_aco.endswith('.xlsx'):
                    aco_source = pd.read_excel(found_aco)
                else:
                    aco_source = pd.read_csv(found_aco)
            else:
                logger.warning("[LSTM] FILE ACO TIDAK DITEMUKAN di path manapun!")

            if aco_source is not None:
                if 'Tanggal' in aco_source.columns:
                    aco_source['Tanggal'] = pd.to_datetime(aco_source['Tanggal'], errors='coerce')
                    aco_source = aco_source.sort_values('Tanggal')
                    
                    if 'Radius_Visual_KM' in aco_source.columns:
                        aco_source['Calculated_Area'] = np.pi * (aco_source['Radius_Visual_KM'] ** 2)
                    
                    aco_renamed = aco_source.rename(columns={
                        'Lintang': 'aco_lat_merged',
                        'Bujur': 'aco_lon_merged',
                        'Radius_Visual_KM': 'aco_radius_merged',
                        'Calculated_Area': 'aco_area_merged'
                    })

                    df = pd.merge_asof(
                        df.sort_values('Acquired_Date'),
                        aco_renamed[['Tanggal', 'aco_lat_merged', 'aco_lon_merged', 'aco_area_merged']].sort_values('Tanggal'),
                        left_on='Acquired_Date',
                        right_on='Tanggal',
                        direction='backward'
                    )

                    for col in ['aco_lat_merged', 'aco_lon_merged', 'aco_area_merged']:
                        if col in df.columns:
                            df[col] = df[col].ffill()

        except Exception as e:
            logger.warning(f"[LSTM] Gagal load data ACO: {e}")

        # =====================================================================
        # BAGIAN 2: LOAD & MERGE DATA GA (Report Data)
        # =====================================================================
        try:
            ga_data_ready = None
            ga_search_paths = [
                r"output/ga_results/ga_report.xlsx",
                os.path.join(out_root, "ga_report.xlsx"),
                "ga_report.xlsx"
            ]

            found_ga = None
            for p in ga_search_paths:
                norm_p = os.path.normpath(p)
                if os.path.exists(norm_p):
                    found_ga = norm_p
                    break
            
            if found_ga:
                logger.info(f"[LSTM] Membaca GA Report dari: {found_ga}")
                try:
                    df_raw_ga = pd.read_excel(found_ga, sheet_name='RawData')
                    df_out_ga = pd.read_excel(found_ga, sheet_name='GA_Output')
                    ga_data_ready = pd.concat([df_raw_ga, df_out_ga], axis=1)
                except Exception as sub_e:
                    logger.warning(f"[LSTM] Gagal baca sheet Excel GA: {sub_e}")

            if ga_data_ready is not None and 'Tanggal' in ga_data_ready.columns:
                ga_data_ready['Tanggal'] = pd.to_datetime(ga_data_ready['Tanggal'], errors='coerce')
                ga_data_ready = ga_data_ready.sort_values('Tanggal')
                
                rename_map = {}
                if 'angle_deg' in ga_data_ready.columns: rename_map['angle_deg'] = 'ga_angle_merged'
                if 'distance_km' in ga_data_ready.columns: rename_map['distance_km'] = 'ga_dist_merged'
                
                ga_ready = ga_data_ready.rename(columns=rename_map)
                cols_to_merge = ['Tanggal'] + list(rename_map.values())
                
                df = pd.merge_asof(
                    df.sort_values('Acquired_Date'),
                    ga_ready[cols_to_merge].sort_values('Tanggal'),
                    left_on='Acquired_Date',
                    right_on='Tanggal',
                    direction='backward'
                )

                for col in ['ga_angle_merged', 'ga_dist_merged']:
                    if col in df.columns:
                        df[col] = df[col].ffill()

        except Exception as e:
            logger.warning(f"[LSTM] Gagal load data GA: {e}")

        # =====================================================================
        # BAGIAN 3: EXPORT KHUSUS UNTUK CNN ENGINE (MODIFIKASI PENTING DISINI)
        # =====================================================================
        try:
            # Kita buat copy khusus agar tidak mengganggu format Client
            df_for_cnn = df.copy()

            # Helper untuk mengambil value prioritas (Merge > JSON > Kolom Asli)
            def get_val_raw(targets, default=0.0):
                for t in targets:
                    if t in df_for_cnn.columns:
                        return df_for_cnn[t].fillna(default)
                return default

            # 1. Pastikan kolom yang dibutuhkan TabularFeatureExtractor (CNN) tersedia
            # Mapping ke nama variabel yang dimengerti CNN Engine
            df_for_cnn['aco_center_lat'] = get_val_raw(['aco_lat_merged', 'aco_center_lat', 'Lintang'])
            df_for_cnn['aco_area_km2']   = get_val_raw(['aco_area_merged', 'aco_area_km2', 'Area'])
            
            # Anomaly Score / LSTM Prediction (Input node ke-5 CNN)
            if 'anomaly_score' not in df_for_cnn.columns:
                df_for_cnn['anomaly_score'] = 0.0
            
            # Target Calculation di CNN butuh koordinat EQ asli
            if 'EQ_Lintang' not in df_for_cnn.columns: df_for_cnn['EQ_Lintang'] = 0.0
            if 'EQ_Bujur' not in df_for_cnn.columns: df_for_cnn['EQ_Bujur'] = 0.0
            if 'cluster_id' not in df_for_cnn.columns: df_for_cnn['cluster_id'] = -1

            # Select Columns spesifik untuk CNN
            cnn_cols = [
                'Acquired_Date', 'EQ_Lintang', 'EQ_Bujur', 'cluster_id',
                'aco_center_lat', 'aco_area_km2', 'anomaly_score'
            ]
            
            path_cnn_input = os.path.join(out_root, "cnn_input_data.csv")
            df_for_cnn[cnn_cols].to_csv(path_cnn_input, index=False)
            logger.info(f"[LSTM] >>> FILE KHUSUS CNN DIBUAT: {path_cnn_input}")
            
        except Exception as e:
            logger.error(f"[LSTM] Gagal membuat file input CNN: {e}")


        # =====================================================================
        # BAGIAN 4: FINAL MAPPING & SAVING UNTUK CLIENT (FORMAT DATA LAMA/BARU)
        # =====================================================================
        
        # Persiapan Kolom Anomali
        df['Anomaly_Status'] = 'Normal'
        if anomalies is not None and not anomalies.empty:
            df.loc[df.index.isin(anomalies.index), 'Anomaly_Status'] = 'Anomaly'

        df['Waktu'] = df['Acquired_Date']

        # Helper Mapping Client (Bahasa Indonesia/User Friendly)
        def get_val(targets, default=0.0):
            for t in targets:
                if t in df.columns:
                    return df[t].fillna(default)
            return default

        def deg_to_compass(deg):
            try:
                if pd.isna(deg): return "Tidak diketahui"
                deg = deg % 360
                directions = ["Utara", "Timur Laut", "Timur", "Tenggara", "Selatan", "Barat Daya", "Barat", "Barat Laut"]
                idx = int((deg + 22.5) // 45) % 8
                return directions[idx]
            except: return "Tidak diketahui"

        # Mapping Client Output
        df['ACO_Pusat_Lat'] = get_val(['aco_lat_merged', 'aco_center_lat', 'Lintang']) #output lstm
        df['ACO_Pusat_Lon'] = get_val(['aco_lon_merged', 'aco_center_lon', 'Bujur'])
        df['ACO_Area']      = get_val(['aco_area_merged', 'aco_area_km2', 'Area'])
        
        df['GA_Sudut']      = get_val(['ga_angle_merged', 'ga_bearing', 'angle_deg'])
        df['GA_Arah_Jarak'] = get_val(['ga_dist_merged', 'ga_distance_km', 'distance_km'])
        df['GA_Arah'] = df['GA_Sudut'].apply(deg_to_compass)

        df['Anomali'] = df['Anomaly_Status']
       
        # Final Select 8 Kolom Client
        final_cols = ['Waktu', 'ACO_Pusat_Lat', 'ACO_Pusat_Lon', 'ACO_Area',
                      'GA_Sudut', 'GA_Arah', 'GA_Arah_Jarak', 'Anomali']

        for c in final_cols:
            if c not in df.columns: df[c] = 0.0
            
        df_final = df[final_cols].copy()

        # Splitting berdasarkan tahun
        mask_old = (df_final['Waktu'].dt.year >= 2022) & (df_final['Waktu'].dt.year <= 2024)
        df_old = df_final.loc[mask_old]

        mask_new = df_final['Waktu'].dt.year == 2025
        df_new = df_final.loc[mask_new]

        try:
            path_old = os.path.join(out_root, "data_lama_2022_2024.csv")
            path_new = os.path.join(out_root, "data_baru_2025.csv")

            df_old.to_csv(path_old, index=False)
            df_new.to_csv(path_new, index=False)
            
            logger.info(f"[LSTM] Data Client saved successfully.")
            logger.info(f"   1. {path_old} (Rows: {len(df_old)})")
            logger.info(f"   2. {path_new} (Rows: {len(df_new)})")

            return {"old": path_old, "new": path_new}

        except Exception as e:
            logger.error(f"[LSTM] Save failed: {e}")
            return {}


    # fungsi prediksi realtime (dummy untuk kompatibilitas)
    def predict_realtime(self, *args):
        return pd.DataFrame(), pd.DataFrame()
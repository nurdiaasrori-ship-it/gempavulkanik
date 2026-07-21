# main.py (Versi V5.2 - Excel Storage Edition + ACO Activated First - FIXED FINAL)
# -- coding: utf-8 --

# Di awal main.py
import warnings #untuk mengontrol peringatan (warning) yang dikeluarkan program.
# Mengabaikan FutureWarning Pandas dan RuntimeWarning NumPy yang umum
warnings.filterwarnings("ignore", category=FutureWarning) # untuk menyembunyikan pesan peringatan yang tidak kritis
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Non-aktifkan pesan diagnostik internal TF yang sudah usang
import os # Mengatur variabel lingkungan sistem operasi.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os 
import sys # untuk menambahkan folder proyek ke jalur pencarian Python
import time # Menghitung durasi proses (uptime), atau melakukan jeda (time.sleep) saat looping pemantauan realtime.
import json # Membaca/menyimpan konfigurasi, menyimpan state pipeline, atau menyimpan hasil prediksi agar bisa dibaca oleh Dashboard/Web.
import logging # Mencatat aktivitas program (logging)
import argparse # Memungkinkan menjalankan program dengan opsi tambahan, misal: python main.py --skip-training
import platform # untuk penyesuaian path file yang berbeda antar OS
import traceback #  Mencetak stack trace saat error terjadi.
import functools # igunakan untuk @functools.wraps dalam decorator pipeline_guard, agar metadata fungsi asli tidak hilang saat dibungkus wrapper anti-crash.
from datetime import datetime # Manipulasi tanggal dan jam. 
from typing import Dict, Optional, Tuple, Any, List # Membantu developer (dan IDE) memahami tipe data apa yang diharapkan (misal: List, Dict, Optional) agar kode lebih rapi dan minim bug.

import pandas as pd # Analisis dan manipulasi data tabular.
import numpy as np # untuk operasi matematika vektor, perhitungan jarak Haversine, atau manipulasi array sebelum masuk ke Neural Network.
from sklearn.model_selection import train_test_split # Membagi dataset
from VolcanoAI.postprocess.cnn_csv_to_json import run as cnn_csv_to_json # Mengonversi hasil prediksi CNN (CSV) menjadi format JSON agar mudah dibaca oleh peta interaktif atau dashboard web.
from VolcanoAI.engines.cnn_map_generator import CNNMapGenerator # Membuat peta visualisasi (biasanya file HTML) berdasarkan hasil prediksi spasial CNN.
from VolcanoAI.utils.presentation_exporter import generate_presentation_excel

# ==============================================================================
# 1. SYSTEM INITIALIZATION & PATH SETUP
# ==============================================================================

def setup_project_path(): 
    """Memastikan root project terdaftar di sys.path."""
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        project_root = os.path.abspath('.')
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parent_dir = os.path.dirname(project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

setup_project_path()

# ==============================================================================
# 2. MODULE IMPORTS (SAFE LOADER)
# ==============================================================================

SYSTEM_READY = False

try:
    from VolcanoAI.config.config import CONFIG, ProjectConfig # Memuat konfigurasi proyek dari file config.py
    from VolcanoAI.utils.setup_logging import setup_logging # Mengatur logging ke file dan console
    
    from VolcanoAI.processing.data_loader import DataLoader # Memuat dan menggabungkan data dari berbagai sumber (BMKG, Mirova, Excel Injection)
    from VolcanoAI.processing.feature_engineer import FeatureEngineer, FeaturePreprocessor # Melakukan feature engineering pada data (scaling, encoding, rolling stats, dll)
    
    # Realtime manager baru (BMKG + Mirova + Injection Excel)
    from VolcanoAI.processing.realtime_sensor_manager import RealtimeSensorManager # Mengelola pengambilan data realtime dari berbagai sumber sensor
    from VolcanoAI.processing.realtime_buffer_manager import RealtimeBufferManager # Mengelola buffer data realtime untuk inference

    from VolcanoAI.engines.aco_engine import DynamicAcoEngine # Mesin ACO untuk optimasi jalur dampak vulkanik
    from VolcanoAI.engines.ga_engine import GaEngine # Mesin GA untuk optimasi jalur evakuasi
    from VolcanoAI.engines.lstm_engine import LstmEngine # Mesin LSTM untuk prediksi deret waktu gempa
    from VolcanoAI.engines.cnn_engine import CnnEngine # Mesin CNN untuk prediksi spasial gempa
    from VolcanoAI.engines.naive_bayes_engine import NaiveBayesEngine # Mesin Naive Bayes untuk klasifikasi akhir
    from VolcanoAI.engines.cnn_map_generator import CNNMapGenerator # Menghasilkan peta interaktif dari hasil prediksi CNN
    from VolcanoAI.reporting.comprehensive_reporter import ComprehensiveReporter, GraphVisualizer # Laporan komprehensif dan visualisasi hasil

    SYSTEM_READY = True # untuk memastikan apakah semua modul berhasil diload

except ImportError as e: # 
    print(f"\n[CRITICAL IMPORT ERROR] {e}")
    print("Sistem tidak dapat dijalankan karena modul hilang.\n")


# ==============================================================================
# 3. GLOBAL CONSTANTS & HELPER METHODS
# ==============================================================================

SYSTEM_VERSION = "5.2.0-TITAN-EXCEL" # Konstanta global untuk menandai versi sistem saat ini

def pipeline_guard(func): # Decorator: Fungsi pembungkus untuk menangani error otomatis
    @functools.wraps(func) # Menyalin metadata (nama/docstring) fungsi asli ke wrapper
    def wrapper(*args, **kwargs): # Fungsi pengganti yang menerima argumen apa saja
        logger = logging.getLogger("PipelineGuard") if SYSTEM_READY else None # logger jika sistem siap
        try:
            return func(*args, **kwargs) # coba jalankan fungsi asli yang dilindungi
        except Exception as e: 
            if logger:
                logger.critical(f"CRASH di {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
            else:
                print(f"CRASH di {func.__name__}: {str(e)}")
            raise e
    return wrapper


class SystemHealthMonitor: # Kelas untuk memantau kesehatan hardware (RAM/CPU)
    @staticmethod
    def check_resources():
        if not SYSTEM_READY:
            return
        logger = logging.getLogger("SystemMonitor")
        try:
            import psutil # Import library pemantau sistem (jika terinstall)
            mem = psutil.virtual_memory() # Ambil statistik memori (RAM) saat ini
            logger.info(f"RAM Usage: {mem.percent}%") # Catat persentase pemakaian RAM ke log
        except ImportError: # Tangkap error jika library 'psutil' tidak ada
            pass

class PipelineStateManager: # Kelas untuk mencatat status tahapan pipeline ke file JSON
    def __init__(self, output_dir: str):
        self.state_file = os.path.join(output_dir, "pipeline_state.json") # Tentukan lokasi file 'pipeline_state.json' di folder output

    def update_stage(self, stage_name: str, status: str): # Update status tahapan tertentu
        state = self.load_state() # Baca dulu state yang sudah ada
        state[stage_name] = {"status": status, "timestamp": datetime.now().isoformat()}
        try:
            with open(self.state_file, "w") as f: # Buka file JSON mode tulis (write)
                json.dump(state, f, indent=4) # Simpan dictionary ke file JSON dengan indentasi
        except Exception:
            pass

    def load_state(self) -> Dict: # Membaca file status pipeline
        if os.path.exists(self.state_file): # Cek apakah file JSON sudah ada
            try:
                with open(self.state_file, "r") as f: # Buka file mode baca (read)
                    return json.load(f) # Parsing isi JSON menjadi Dictionary
            except Exception: # Jika file rusak/korup
                return {} # Kembalikan dictionary kosong
        return {} # Kembalikan kosong jika file belum ada


# ======================================================================
# 4. SAFE LOAD / SAVE CSV (GLOBAL — WAJIB DI LUAR CLASS)
# ======================================================================

def safe_load_csv(path: str):  # Fungsi untuk membaca file CSV dengan penanganan error
    try:
        if os.path.exists(path): # Cek apakah file benar-benar ada di folder
            return pd.read_csv(path, parse_dates=["Acquired_Date"])
    except Exception as e:
        logging.error(f"[SAFE LOAD ERROR] {path} → {e}")
    return pd.DataFrame()


def safe_save_csv(path: str, df: pd.DataFrame):# Fungsi untuk menyimpan DataFrame ke file CSV
    try:
        df.to_csv(path, index=False)
        logging.info(f"[BUFFER SAVED] {path}")
    except Exception as e:
        logging.error(f"[SAFE SAVE ERROR] {path} → {e}")


# ==============================================================================
# 5. MAIN PIPELINE CLASS
# ==============================================================================

class VolcanoAiPipeline: # Kelas utama pengatur seluruh alur kerja AI
    def __init__(self, config): # Constructor: Dijalankan pertama kali saat class dipanggil
        if not SYSTEM_READY: # Cek flag global apakah semua library berhasil di-import
            return

        self.config = config # Simpan konfigurasi project ke variabel class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_mgr = PipelineStateManager(self.config.OUTPUT.directory)
        # Inisialisasi variabel penampung data (masih kosong)
        self.df_train = None # Tempat data latih
        self.df_test = None # Tempat data uji
        self.feature_preprocessor = None # Tempat preprocessor hasil FE
        # Inisialisasi variabel penampung Engine AI (masih kosong)
        self.trained_aco_engine = None # Engine Ant Colony (Risk Zoning)
        self.trained_ga_engine = None # Engine GA (Evacuation Path)
        self.trained_lstm_engine = None # Engine LSTM (Time Series Prediction)
        self.trained_cnn_engine = None # Engine CNN (Spatial Prediction)
        self.trained_nb_engine = None # Engine Naive Bayes (Final Classifier)

        self._init_subsystems() # Panggil fungsi untuk menyiapkan/mengisi engine di atas
        self._init_paths() # Panggil fungsi untuk menyiapkan path cache data

        self.logger.info(f"Pipeline VolcanoAI {SYSTEM_VERSION} initialized successfully.")

    # ----------------------------------------------------------------------
    # INIT SYSTEM
    # ----------------------------------------------------------------------

    def _init_subsystems(self): # Fungsi privat untuk menginstansiasi semua sub-modul
        self.data_loader = DataLoader(self.config.DATA_LOADER) # Siapkan modul pemuat data
        self.feature_engineer = FeatureEngineer(self.config.FEATURE_ENGINEERING, self.config.ACO_ENGINE) # Siapkan modul rekayasa fitur (Feature Engineering)
        # Siapkan Engine AI satu per satu menggunakan config masing-masing
        self.aco_engine = DynamicAcoEngine(self.config.ACO_ENGINE) # Engine ACO
        self.ga_engine = GaEngine(self.config.GA_ENGINE) # Engine GA
        self.lstm_engine = LstmEngine(self.config.LSTM_ENGINE) # Engine LSTM
        self.cnn_engine = CnnEngine(self.config.CNN_ENGINE) # Engine CNN
        self.nb_engine = NaiveBayesEngine(self.config.NAIVE_BAYES_ENGINE) # Engine Naive Bayes

        self.reporter = ComprehensiveReporter(self.config) # Siapkan modul pelaporan komprehensif

        # Realtime Sensor Manager (BMKG + MIROVA + Injection Excel)
        # Sesuaikan path log MIROVA kalau perlu
        self.sensor_manager = RealtimeSensorManager(
            mirova_log_path="output/realtime/" # lokasi log Mirova disimpan
        )

    def _init_paths(self): # Fungsi untuk menyiapkan path (lokasi file) cache
        self.data_cache_path = self.config.DATA_LOADER.merged_output_path.replace(".xlsx", ".pkl") # Path cache data gabungan (pickle)
        self.preprocessor_cache_path = self.config.FEATURE_ENGINEERING.preprocessor_output_path # Path cache preprocessor FE


    # ----------------------------------------------------------------------
    # PHASE 1 — LOAD & SPLIT DATA
    # ----------------------------------------------------------------------

    @pipeline_guard # Decorator untuk menangani error otomatis
    def _step_load_and_split_data(self) -> bool: # Fungsi untuk memuat dan membagi data
        self.logger.info("\n========== PHASE 1: DATA LOADING ==========")
        SystemHealthMonitor.check_resources() # Cek kesehatan sistem (RAM/CPU)
        self.state_mgr.update_stage("DataLoading", "Running")

        df_full = None # Tempat data gabungan penuh
        cache_exists = os.path.exists(self.data_cache_path) # Cek apakah file cache data gabungan sudah ada

        if self.config.PIPELINE.run_data_loading: # Cek apakah konfigurasi mengizinkan pemuatan data
            if not cache_exists: # Jika cache belum ada, lakukan pemuatan data penuh
                df_full = self.data_loader.run() # Panggil modul pemuat data untuk mendapatkan data gabungan
                if df_full is None or df_full.empty: # Cek apakah data gabungan kosong
                    return False # Jika kosong, hentikan proses dan kembalikan False
                from VolcanoAI.processing.preprocess_eq import preprocess_earthquake_data # Import fungsi pra-pemrosesan data gempa
                df_full = preprocess_earthquake_data(df_full) # Lakukan pra-pemrosesan pada data gabungan
                df_full.to_pickle(self.data_cache_path) # Simpan data gabungan ke file cache (pickle)
            else: # Jika cache sudah ada, muat data dari file cache
                df_full = pd.read_pickle(self.data_cache_path) # Baca data gabungan dari file cache (pickle)

        if df_full is None or df_full.empty: # Cek lagi apakah data gabungan kosong
            return False # Jika kosong, hentikan proses dan kembalikan False

        # tidak diacak/displit! 
        # Paksa seluruh data masuk ke df_train agar diekspor ke cnn_predictions_latest.csv
        self.df_train = df_full.copy()
        self.df_test = df_full.copy()

        self.state_mgr.update_stage("DataLoading", "Success") # Update status tahapan pemuatan data menjadi sukses
        return True # Kembalikan True menandakan proses berhasil


    # ----------------------------------------------------------------------
    # PHASE 2 — FEATURE ENGINEERING
    # ----------------------------------------------------------------------

    @pipeline_guard # Decorator untuk menangani error otomatis
    def _step_feature_engineering(self, is_training: bool) -> bool: # Fungsi untuk melakukan rekayasa fitur (feature engineering)
        self.logger.info("\n========== PHASE 2: FEATURE ENGINEERING ==========")
        self.state_mgr.update_stage("FeatureEngineering", "Running") # Update status tahapan FE menjadi berjalan

        if is_training: # Jika dalam mode pelatihan
            self.df_train, self.feature_preprocessor = self.feature_engineer.run(self.df_train) # Lakukan FE pada data latih dan dapatkan preprocessor

            if self.df_test is not None: # Jika data uji ada
                self.df_test, _ = self.feature_engineer.run(self.df_test, preprocessor=self.feature_preprocessor) # Lakukan FE pada data uji menggunakan preprocessor dari data latih
                 
        else: # Jika dalam mode inferensi (bukan pelatihan)
            if self.feature_preprocessor is None: # Jika preprocessor belum ada
                if os.path.exists(self.preprocessor_cache_path): # Cek apakah file cache preprocessor sudah ada
                    self.feature_preprocessor = FeaturePreprocessor.load(self.preprocessor_cache_path) # Muat preprocessor dari file cache
                else: # Jika file cache preprocessor tidak ada
                    return False # Hentikan proses dan kembalikan False

            if self.df_test is not None: # Jika data uji ada
                self.df_test, _ = self.feature_engineer.run(self.df_test, preprocessor=self.feature_preprocessor) # Lakukan FE pada data uji menggunakan preprocessor yang dimuat
            if self.df_train is not None: # Jika data latih ada
                self.df_train, _ = self.feature_engineer.run(self.df_train, preprocessor=self.feature_preprocessor) # Lakukan FE pada data latih menggunakan preprocessor yang dimuat

        self.state_mgr.update_stage("FeatureEngineering", "Success") # Update status tahapan FE menjadi sukses
        return True # Kembalikan True menandakan proses berhasil


    # ----------------------------------------------------------------------
    # PHASE 3 — TRAINING FLOW (FINAL VERSION)
    # ----------------------------------------------------------------------

    @pipeline_guard # Decorator untuk menangani error otomatis
    def _run_training_flow(self):
        self.logger.info("\n========== PHASE 3: MODEL TRAINING ==========")

        # =========================
        # INIT DATA
        # =========================
        if self.df_train is None or self.df_train.empty:
            self.logger.critical("[PIPELINE] df_train kosong / None → training dibatalkan")
            self.trained_nb_engine = None
            return

        df_processed = self.df_train.copy()

        # =========================
        # 1️⃣ ACO
        # =========================
        try:
            df_processed, _ = self.aco_engine.run(df_processed)
            self.trained_aco_engine = self.aco_engine
        except Exception as e:
            self.logger.exception(f"[ACO] failed: {e}")
            return

        # =========================
        # 2️⃣ GA
        # =========================
        try:
            self.ga_engine.run(df_processed)
            self.trained_ga_engine = self.ga_engine
        except Exception as e:
            self.logger.exception(f"[GA] failed: {e}")
            return

        # =========================
        # 3️⃣ LSTM
        # =========================
        try:
            self.lstm_engine.train(df_processed)
            self.trained_lstm_engine = self.lstm_engine

            df_processed, _ = self.lstm_engine.predict_on_static(df_processed)
            if df_processed is None or df_processed.empty:
                raise ValueError("LSTM returned None / empty DataFrame")
        except Exception as e:
            self.logger.exception(f"[LSTM] failed → pipeline stopped: {e}")
            return

        # =========================
        # 4️⃣ CNN
        # =========================
        try:
            self.cnn_engine.train(df_processed, self.lstm_engine)
            self.trained_cnn_engine = self.cnn_engine

            df_cnn = self.cnn_engine.predict(df_processed, self.lstm_engine)
            if df_cnn is None or not hasattr(df_cnn, "columns") or df_cnn.empty:
                raise ValueError("CNN returned invalid DataFrame")
            df_processed = df_cnn
        except Exception as e:
            self.logger.warning(
                f"[CNN] training/predict failed - applying fallback. Error: {e}"
            )
            self.trained_cnn_engine = None

            # =========================
            # SAFE FALLBACK AREA
            # =========================
            try:
                divisor = getattr(self.config.CNN_ENGINE, "luas_unit_divisor", 1.0)
                divisor = float(divisor) if divisor else 1.0
            except Exception:
                divisor = 1.0

            if 'AreaTerdampak_km2' in df_processed.columns:
                df_processed['luas_cnn'] = (
                    df_processed['AreaTerdampak_km2']
                    .fillna(0.0)
                    .astype(float)
                    / divisor
                )
            else:
                df_processed['luas_cnn'] = 0.0

            df_processed['cnn_confidence'] = 0.25


        # =========================
        # CNN ERROR CHECK
        # =========================
        if hasattr(self.cnn_engine, "evaluate_error"):
            try:
                cnn_error = self.cnn_engine.evaluate_error(df_processed)
                self.logger.info(f"[CNN] Training error: {cnn_error:.4f}")

                if cnn_error > self.config.CNN_ENGINE.max_error_threshold:
                    self.logger.warning("[CNN] Error tinggi → retraining CNN")
                    self.cnn_engine.train(df_processed, self.lstm_engine)
            except Exception as e:
                self.logger.warning(f"[CNN] error evaluation skipped: {e}")

        # =========================
        # EXPORT RESULTS
        # =========================
        from datetime import datetime
        from pathlib import Path

        out_dir = Path(self.config.OUTPUT.directory) / "cnn_results" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        export_cols = [
            "Acquired_Date",      # waktu
            "cnn_angle_deg",      # sudut
            "cnn_cardinal",       # arah
            "cnn_distance_km",    # distance
        ]

        export_cols = [c for c in export_cols if c in df_processed.columns]

        if not export_cols:
            self.logger.critical("[EXPORT] Tidak ada kolom valid untuk diekspor → STOP")
            return

        latest_path = out_dir / "cnn_predictions_latest.csv"
        df_processed[export_cols].to_csv(latest_path, index=False)
        self.logger.info(f"✅ CNN latest overwritten: {latest_path}")

        # =========================
        # PRESENTATION EXCEL EXPORT (CLIENT)
        # =========================
        try:
            presentation_dir = (
                Path(self.config.OUTPUT.directory)
                / "cnn_results"
                / "presentation"
            )

            excel_path = generate_presentation_excel(
                csv_latest_path=latest_path,
                output_dir=presentation_dir
            )

            self.logger.info(f"📊 Presentation Excel generated: {excel_path}")

        except Exception as e:
            self.logger.exception(f"[PRESENTATION EXCEL] failed: {e}")

        # =========================
        # JSON EXPORT
        # =========================
        try:
            cnn_json_path = (
                Path(self.config.OUTPUT.directory)
                / "cnn_results"
                / "cnn_predictions_latest.json"
            )

            cnn_csv_to_json(
                csv_path=str(latest_path),
                out_json=str(cnn_json_path),
                force=True,
            )
            self.logger.info(f"✅ CNN JSON generated: {cnn_json_path}")
        except Exception as e:
            self.logger.exception(f"[CNN JSON] failed: {e}")

        # =========================
        # MAP GENERATION
        # =========================
        try:
            cnn_output_dir = Path(self.config.OUTPUT.directory) / "cnn_results" / "maps"

            if cnn_json_path.exists():
                cnn_map_gen = CNNMapGenerator(output_dir=cnn_output_dir)
                map_path = cnn_map_gen.generate(json_path=cnn_json_path)

                if map_path:
                    self.logger.info(f"🗺️ CNN map generated: {map_path}")
        except Exception as e:
            self.logger.exception(f"[CNN MAP] failed: {e}")

        # =========================
        # EMPIRICAL ACTIVATION PLOT
        # =========================
        try:
            from VolcanoAI.visualization.generate_activation_plot import generate_relu_activation_plot
            self.logger.info("📊 Membuat grafik empiris aktivasi CNN ReLU...")
            plot_path = generate_relu_activation_plot(cluster_id=0)
            if plot_path:
                self.logger.info(f"✅ Grafik Aktivasi CNN disimpan di: {plot_path}")
        except Exception as e:
            self.logger.warning(f"[CNN PLOT] Pembuatan grafik aktivasi dilewati: {e}")

        # =========================
        # ARCHIVE
        # =========================
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = out_dir / f"cnn_predictions_{ts}.csv"
        df_processed[export_cols].to_csv(archive_path, index=False)
        self.logger.info(f"📦 CNN archive saved: {archive_path}")

        # =========================
        # 5️⃣ NAIVE BAYES (FIX UTAMA)
        # =========================
        try:
            nb_ok = self.nb_engine.train(df_processed)
            if not nb_ok:
                raise RuntimeError("Naive Bayes training gagal (return False)")
            self.trained_nb_engine = self.nb_engine
        except Exception as e:
            self.trained_nb_engine = None
            self.logger.exception(f"[NB] failed → engine disabled: {e}")
            return

        # =========================
        # FINAL STATE
        # =========================
        self.state_mgr.update_stage("Training", "Success")

    # ----------------------------------------------------------------------
    # PHASE 4 — EVALUATION
    # ----------------------------------------------------------------------

    @pipeline_guard
    def _run_evaluation_flow(self):
        self.logger.info("\n========== PHASE 4: MODEL EVALUATION ==========")

        # =========================
        # INIT DATA
        # =========================
        if self.df_test is None or self.df_test.empty:
            self.logger.critical("[EVAL] df_test kosong / None → evaluasi dibatalkan")
            return None, {}, []

        df_eval = self.df_test.copy()

        # =========================
        # ACO (DYNAMIC)
        # =========================
        try:
            df_eval, _ = DynamicAcoEngine(self.config.ACO_ENGINE).run(df_eval)
        except Exception as e:
            self.logger.exception(f"[EVAL][ACO] failed: {e}")
            return df_eval, {}, []

        # =========================
        # LOAD ENGINES
        # =========================
        lstm = self.trained_lstm_engine
        cnn  = self.trained_cnn_engine
        nb   = self.trained_nb_engine

        # =========================
        # LSTM SAFETY
        # =========================
        if lstm is None or not hasattr(lstm, "predict_on_static"):
            self.logger.critical("[EVAL] LSTM engine invalid / None → STOP")
            return df_eval, {}, []

        df_eval, anomalies = lstm.predict_on_static(df_eval)

        # =========================
        # CNN SAFETY
        # =========================
        if cnn is None or not hasattr(cnn, "predict"):
            self.logger.warning("[EVAL] CNN engine invalid / None → skip CNN")
        else:
            df_eval = cnn.predict(df_eval, lstm)

        # =========================
        # NAIVE BAYES SAFETY (FIX UTAMA)
        # =========================
        if nb is None:
            self.logger.critical("[EVAL] Naive Bayes engine = None → evaluasi dilewati")
            self.state_mgr.update_stage("Evaluation", "Skipped")
            return df_eval, {}, anomalies

        if not hasattr(nb, "evaluate"):
            self.logger.critical("[EVAL] Naive Bayes engine tidak punya evaluate()")
            self.state_mgr.update_stage("Evaluation", "Failed")
            return df_eval, {}, anomalies

        # =========================
        # NAIVE BAYES EVALUATION
        # =========================
        try:
            df_final, metrics = nb.evaluate(df_eval)
        except Exception as e:
            self.logger.exception(f"[EVAL][NB] gagal evaluate: {e}")
            self.state_mgr.update_stage("Evaluation", "Failed")
            return df_eval, {}, anomalies

        # =========================
        # SUCCESS
        # =========================
        self.state_mgr.update_stage("Evaluation", "Success")
        return df_final, metrics, anomalies


    # ----------------------------------------------------------------------
    # PHASE X — REALTIME INFERENCE (BUFFER + INJECTION + FE)
    # ----------------------------------------------------------------------

    @pipeline_guard # Decorator untuk menangani error otomatis
    def run_realtime_inference(self): # Fungsi untuk menjalankan inferensi realtime
        self.logger.info("\n========== REALTIME INFERENCE MODE ==========") 

        buffer = RealtimeBufferManager(buffer_days=90) # Inisialisasi manajer buffer realtime dengan kapasitas 90 hari

        # Load buffer lama (kalau ada)
        buffer.raw_realtime = safe_load_csv("output/realtime/raw_realtime.csv") # Muat data mentah realtime dari file CSV
        buffer.raw_injection = safe_load_csv("output/realtime/raw_injection.csv") # Muat data mentah injection dari file CSV
        buffer.processed = safe_load_csv("output/realtime/processed.csv") # Muat data terproses dari file CSV

        # Ambil data realtime dari sensor manager (BMKG + Mirova + Injection)
        try:
            # [FIX 1]: Mengganti get_realtime_data() dengan get_merged_stream()
            df_raw_new_stream = self.sensor_manager.get_merged_stream() 
        except Exception as e:
            self.logger.error(f"Realtime fetch error: {e}")
            df_raw_new_stream = pd.DataFrame() # Mengganti 3 df output menjadi 1

        # Pisahkan BMKG dan Injection sebelum append (agar data mentah tersimpan terpisah)
        # Asumsi: df_raw_new_stream mengandung kedua sumber, dan kita perlu memisahkannya 
        df_bmkg = df_raw_new_stream[df_raw_new_stream['Sumber'] == 'BMKG'].copy() 
        df_inj = df_raw_new_stream[df_raw_new_stream['Sumber'] == 'InjectedExcel'].copy()

        # Simpan BMKG ke buffer realtime
        if df_bmkg is not None and not df_bmkg.empty: # Cek apakah data BMKG tidak kosong
            buffer.append_raw_realtime(df_bmkg) # Tambahkan data BMKG ke buffer realtime

        # Simpan data suntik ke buffer injection
        if df_inj is not None and not df_inj.empty: # Cek apakah data injection tidak kosong
            buffer.append_raw_injection(df_inj) # Tambahkan data injection ke buffer injection

        # df_raw sekarang adalah gabungan dari data mentah baru dan data mentah historis
        df_raw = buffer.get_merged_raw() # Gabungkan data mentah realtime dan injection dari buffer
        if df_raw.empty: # Cek apakah data mentah gabungan kosong
            self.logger.warning("Tidak ada data inference.") # Catat peringatan ke log
            return # Hentikan proses jika tidak ada data untuk inferensi

        # FE untuk realtime (pakai preprocessor hasil training)
        df_processed, _ = self.feature_engineer.run(
            df_raw,
            is_training=False,
            preprocessor=self.feature_preprocessor,
        ) # Lakukan feature engineering pada data mentah gabungan menggunakan preprocessor yang sudah dilatih

        buffer.append_processed(df_processed) # Tambahkan data terproses ke buffer

        # Simpan balik buffer ke disk
        safe_save_csv("output/realtime/raw_realtime.csv", buffer.raw_realtime) # Simpan data mentah realtime ke file CSV
        safe_save_csv("output/realtime/raw_injection.csv", buffer.raw_injection) # Simpan data mentah injection ke file CSV
        safe_save_csv("output/realtime/processed.csv", buffer.processed) # Simpan data terproses ke file CSV

        self.logger.info("Realtime inference complete.") # Catat keberhasilan inferensi realtime ke log


    # ----------------------------------------------------------------------
    # PHASE 5 — LIVE MONITORING LOOP
    # ----------------------------------------------------------------------

    def start_monitoring_loop(self): # Fungsi untuk memulai loop pemantauan realtime
        self.logger.info("\n========== PHASE 5: LIVE MONITORING ==========")

        if self.df_train is not None: # Pastikan data latih sudah ada
            self.lstm_engine.load_buffer(self.df_train) # Muat buffer LSTM dari data latih
             
        interval = self.config.REALTIME.check_interval_seconds # Ambil interval pengecekan dari konfigurasi      

        try: 
            while True: 
                # 1. AMBIL SEMUA DATA MENTAH BARU (BMKG + VRP + INJECT)
                # Dapatkan data mentah dari semua sumber (sudah di-VRP-merge di StreamManager)
                df_mirova_raw, df_bmkg_synced, df_inj_synced = self.sensor_manager.get_realtime_data()
                
                # Gabungkan hanya BMKG dan INJECTION (karena Mirova sudah di-merge)
                frames = []
                if not df_bmkg_synced.empty: frames.append(df_bmkg_synced) # Tambahkan data BMKG jika tidak kosong
                if not df_inj_synced.empty: frames.append(df_inj_synced) # Tambahkan data injection jika tidak kosong
                 
                if not frames: # Jika tidak ada data baru dari kedua sumber
                    time.sleep(interval) # Tunggu sesuai interval sebelum pengecekan berikutnya
                    continue # Lanjutkan ke iterasi berikutnya
                
                df_raw_new_stream = pd.concat(frames, ignore_index=True) # Gabungkan data mentah baru dari kedua sumber
                
                # ----------------------------------------------------
                # 1. GABUNGKAN DENGAN HISTORY UNTUK KONTEKS FE (LAG/ROLLING)
                df_hist = self.lstm_engine.get_buffer() # Ambil data historis dari buffer LSTM
                df_combined_for_fe = pd.concat([df_hist, df_raw_new_stream], ignore_index=True) # Gabungkan data historis dengan data mentah baru untuk konteks FE
                
                # 2. FEATURE ENGINEERING (pada seluruh combined data)
                df_proc, _ = self.feature_engineer.run( 
                    df_combined_for_fe, is_training=False, preprocessor=self.feature_preprocessor 
                ) # Lakukan feature engineering pada data gabungan menggunakan preprocessor yang sudah dilatih
                
                # 3. FILTER HANYA EVENT BARU (TIDAK ADA LAG) & LOKASI
                
                # Filter A: Ambil baris terbaru yang memiliki Acquired_Date > dari data history terlama
                last_hist_date = df_hist['Acquired_Date'].max() if not df_hist.empty and 'Acquired_Date' in df_hist.columns else pd.to_datetime('1900-01-01')
                # Kita harus mencari baris yang memiliki Acquired_Date yang lebih besar dari history TERAKHIR
                df_target_raw = df_proc[df_proc['Acquired_Date'] > last_hist_date].copy()
                
                if df_target_raw.empty: # Jika tidak ada data baru setelah filter tanggal
                    self.logger.info("Tidak ada gempa baru teridentifikasi (atau sudah ada di buffer).") # Catat info ke log
                    time.sleep(interval) # Tunggu sesuai interval sebelum pengecekan berikutnya
                    continue # Lanjutkan ke iterasi berikutnya
                    
                # Filter B: Geografis (Hanya Cluster Aktif)
                valid_clusters = self.lstm_engine.vault.list_clusters() # Dapatkan daftar cluster valid dari LSTM engine
                df_target = df_target_raw[df_target_raw['cluster_id'].isin(valid_clusters)] .copy() # Filter hanya baris dengan cluster_id yang valid
                
                if df_target.empty: # Jika tidak ada data baru setelah filter cluster
                    self.logger.info("Gempa baru terdeteksi, tetapi di luar area fokus yang dilatih. Skip prediksi.") # Catat info ke log
                    time.sleep(interval) # Tunggu sesuai interval sebelum pengecekan berikutnya
                    continue # Lanjutkan ke iterasi berikutnya

                self.logger.info(f"🚨 GEMPA BARU TERDETEKSI: {len(df_target)}") # Catat jumlah gempa baru yang terdeteksi ke log

                # 4. PREDICION FLOW 
                df_pred, anomalies = self.lstm_engine.process_live_stream(df_target) # Proses aliran data baru menggunakan engine LSTM
                df_pred = self.cnn_engine.predict_and_export(df_pred, self.lstm_engine) # Lakukan prediksi menggunakan engine CNN
                df_pred, _ = self.nb_engine.evaluate(df_pred) # Lakukan evaluasi menggunakan engine Naive Bayes

                # =========================
                # SEMI-HYBRID FEEDBACK LOOP
                # =========================

                if 'actual_event' in df_pred.columns: # Cek apakah kolom actual_event ada di DataFrame prediksi
                    df_actual = df_pred[df_pred['actual_event'] == 1].copy() # Filter hanya baris yang merupakan actual event

                    if not df_actual.empty: # Jika ada actual event yang terdeteksi
                        self.logger.info(f"[FEEDBACK] Actual events detected: {len(df_actual)}") # Catat jumlah actual event ke log

                        # 1️⃣ RECORD ACTUAL KE LSTM (INI WAJIB ADA)
                        self.lstm_engine.record_actual_events(df_actual) # Rekam actual event ke dalam engine LSTM

                        # 2️⃣ EVALUASI ERROR CNN
                        if hasattr(self.cnn_engine, "evaluate_error"): # Cek apakah engine CNN memiliki metode evaluasi error
                            cnn_error = self.cnn_engine.evaluate_error(df_actual) # Evaluasi error CNN pada actual event
                            self.logger.info(f"[CNN] Live error: {cnn_error:.4f}") # Catat error CNN pada actual event ke log

                            # 3️⃣ RETRAIN JIKA SALAH
                            if cnn_error > self.config.CNN_ENGINE.max_error_threshold: # Jika error melebihi ambang batas yang ditentukan
                                self.logger.warning("[CNN] Retraining due to live mismatch...") # Catat peringatan ke log
                                self.cnn_engine.train( 
                                    self.lstm_engine.get_buffer(),
                                    self.lstm_engine
                                ) # Latih ulang engine CNN menggunakan buffer LSTM

               # =====================================
                # 5. UPDATE BUFFER (CONFIRMED ONLY)
                # =====================================

                # Update buffer hanya event yang cukup valid
                confirmed_cols = ['actual_event', 'confidence'] # Kolom yang diperlukan untuk validasi

                if all(c in df_pred.columns for c in confirmed_cols): # Cek apakah semua kolom konfirmasi ada di DataFrame prediksi
                    df_confirmed = df_pred[df_pred['confidence'] >= 0.7].copy() # Filter hanya baris dengan confidence >= 0.7
                    self.logger.info(
                        f"[BUFFER] Confirmed events: {len(df_confirmed)} / {len(df_pred)}"
                    ) # Catat jumlah event terkonfirmasi ke log
                else:
                    # Fallback: jika belum ada mekanisme confidence
                    self.logger.warning(
                        "[BUFFER] Kolom confidence belum tersedia → fallback update semua (TEMP)"
                    ) # Catat peringatan ke log
                    df_confirmed = df_pred.copy() # Salin semua baris sebagai fallback

                # Update buffer LSTM dengan data terkonfirmasi
                if not df_confirmed.empty: # Jika ada data terkonfirmasi
                    self.lstm_engine.update_buffer(df_confirmed) # Perbarui buffer LSTM dengan data terkonfirmasi

                # Persist buffer
                df_current_buffer = self.lstm_engine.get_buffer() # Ambil data buffer LSTM saat ini
                safe_save_csv("output/realtime/processed.csv", df_current_buffer) # Simpan data buffer LSTM ke file CSV

                time.sleep(interval) # Tunggu sesuai interval sebelum pengecekan berikutnya


        except KeyboardInterrupt: # Tangkap interupsi keyboard (Ctrl+C)
            self.logger.info("Monitoring dihentikan. Melakukan penyimpanan akhir...") # Catat info penghentian ke log
            
            # Simpan buffer akhir saat dihentikan
            df_final_buffer = self.lstm_engine.get_buffer() # Ambil data buffer LSTM saat ini
            safe_save_csv("output/realtime/processed.csv", df_final_buffer) # Simpan data buffer LSTM ke file CSV
            
            self.logger.info("Penyimpanan buffer live selesai.") # Catat keberhasilan penyimpanan buffer ke log


    # ----------------------------------------------------------------------
    # MAIN PIPELINE RUN WRAPPER
    # ----------------------------------------------------------------------

    def run(self): # Fungsi utama untuk menjalankan seluruh alur kerja pipeline
        start_ts = time.time() # Catat waktu mulai eksekusi
        self.logger.info("Starting Pipeline Execution...") # Catat info mulai eksekusi ke log

        try: # Blok try untuk menangani potensi error selama eksekusi
            if self.config.PIPELINE.run_data_loading: # Cek apakah konfigurasi mengizinkan pemuatan data
                if not self._step_load_and_split_data(): # Panggil fungsi untuk memuat dan membagi data
                    return # Hentikan proses jika gagal

            if self.config.PIPELINE.run_feature_engineering: # Cek apakah konfigurasi mengizinkan rekayasa fitur
                if not self._step_feature_engineering(is_training=True): # Panggil fungsi untuk melakukan rekayasa fitur
                    return # Hentikan proses jika gagal

            if self.config.PIPELINE.run_model_training:  # Cek apakah konfigurasi mengizinkan pelatihan model
                self._run_training_flow() # Panggil fungsi untuk menjalankan alur pelatihan model

            if getattr(self.config.PIPELINE, "run_model_evaluation", True): # Cek apakah konfigurasi mengizinkan evaluasi model
                df_final, metrics, anomalies = self._run_evaluation_flow() # Panggil fungsi untuk menjalankan alur evaluasi model
            else:
                df_final, metrics, anomalies = None, {}, None # Jika tidak, set variabel hasil evaluasi ke None atau kosong


            # 🔥 BARU REPORTING
            if self.config.PIPELINE.run_reporting and df_final is not None: # Cek apakah konfigurasi mengizinkan pelaporan dan hasil evaluasi ada
                from pathlib import Path # manipulasi path file dan direktori
                outdir = Path(self.config.OUTPUT.directory) # Tentukan direktori output utama

                normalized = {} # Tempat penyimpanan metrik terstandarisasi

                aco_json = outdir / "aco_results" / "aco_to_ga.json" # Path ke file JSON hasil ACO
                if aco_json.exists(): # Cek apakah file JSON hasil ACO ada
                    try: # Blok try untuk menangani potensi error saat membaca file JSON
                        j = json.loads(aco_json.read_text(encoding="utf-8")) # Baca dan parse file JSON
                        lat = j.get("center_lat") # Dapatkan nilai latitude dari JSON
                        lon = j.get("center_lon") # Dapatkan nilai longitude dari JSON
                        if lat is not None and lon is not None: # Cek apakah latitude dan longitude tidak None
                            normalized["aco_center"] = f"{lat}, {lon}" # Simpan koordinat pusat ACO dalam format string
                        normalized["aco_area"] = j.get("impact_area_km2") # Simpan area dampak ACO
                        normalized["aco_map"] = str(outdir / "aco_results" / "aco_impact_zones.html") # Simpan path ke peta ACO
                    except Exception: # Jika terjadi error saat membaca atau parsing JSON
                        pass # Abaikan error dan lanjutkan

                    # 2) GA map
                    ga_map = outdir / "ga_results" / "ga_from_aco_map.html" # Path ke file peta hasil GA
                    if ga_map.exists(): # Cek apakah file peta hasil GA ada
                        normalized["ga_map"] = str(ga_map) # Simpan path ke peta GA

                    # 3) LSTM CSV files (look for common filenames) 
                    lstm_dir = outdir / "lstm_results" # Direktori hasil LSTM
                    if lstm_dir.exists(): # Cek apakah direktori hasil LSTM ada 
                        for f in ["lstm_records_2y_20241230.csv", "master.csv"]: # Cari file master LSTM
                            p = lstm_dir / f # Path ke file master LSTM
                            if p.exists(): # Cek apakah file master LSTM ada
                                normalized["lstm_master_csv"] = str(p) # Simpan path ke file master LSTM
                                break # Hentikan pencarian setelah menemukan file pertama yang ada
                        for f in ["lstm_recent_15d_20241230.csv", "recent.csv"]: # Cari file recent LSTM
                            p = lstm_dir / f
                            if p.exists():
                                normalized["lstm_recent_csv"] = str(p)
                                break
                        for f in ["lstm_anomalies_20241230.csv", "anomalies.csv"]: # Cari file anomali LSTM
                            p = lstm_dir / f
                            if p.exists():
                                normalized["lstm_anomalies_csv"] = str(p)
                                break

                    # 4) CNN outputs
                    cnn_latest = outdir / "cnn_results" / "results" / "cnn_predictions_latest.csv" # Path ke file CSV prediksi CNN terbaru
                    if cnn_latest.exists():
                        normalized["cnn_pred_csv"] = str(cnn_latest)
                    cnn_json = outdir / "cnn_results" / "cnn_predictions_latest.json" # Path ke file JSON prediksi CNN terbaru
                    if cnn_json.exists():
                        normalized["cnn_pred_json"] = str(cnn_json) # Simpan path ke file JSON prediksi CNN

                    # 5) NaiveBayes outputs (support multiple candidate dirs & files)
                    nb_candidates = [
                                outdir / "naive_bayes",           # legacy
                                outdir / "naive_bayes_results",
                                outdir / "naive_bayes_outputs"
                    ] # Daftar direktori kandidat untuk hasil Naive Bayes
                    nb_found = None # Tempat penyimpanan direktori hasil Naive Bayes yang ditemukan
                    for d in nb_candidates: # Iterasi melalui direktori kandidat
                        if d.exists(): # Cek apakah direktori kandidat ada
                            nb_found = d # Simpan direktori yang ditemukan
                            break # Hentikan pencarian setelah menemukan direktori pertama yang ada

                    if nb_found: # Jika direktori hasil Naive Bayes ditemukan
                        p1 = nb_found / "confusion_matrix.png" # Path ke file gambar confusion matrix
                        p2 = nb_found / "roc_curves.png" # Path ke file gambar ROC curves
                        p_csv = nb_found / "naive_bayes_predictions.csv" # Path ke file CSV prediksi Naive Bayes
                        p_json = nb_found / "naive_bayes_metrics.json" # Path ke file JSON metrik Naive Bayes
                        if p1.exists(): normalized["nb_confusion_png"] = str(p1) # Simpan path ke file gambar confusion matrix
                        if p2.exists(): normalized["nb_roc_png"] = str(p2) # Simpan path ke file gambar ROC curves
                        if p_csv.exists(): normalized["nb_pred_csv"] = str(p_csv) # Simpan path ke file CSV prediksi Naive Bayes
                        if p_json.exists(): normalized["nb_metrics_json"] = str(p_json) # Simpan path ke file JSON metrik Naive Bayes


                    # 6) combine: normalized keys override only when not present in original metrics
                    final_metrics = {} # Tempat penyimpanan metrik akhir untuk pelaporan
                    if isinstance(metrics, dict): # Pastikan metrics adalah dictionary
                        final_metrics.update(metrics) # Salin metrik asli ke metrik akhir
                    # set defaults where missing 
                    for k, v in normalized.items(): # Iterasi melalui metrik terstandarisasi
                        final_metrics.setdefault(k, v) # Set metrik terstandarisasi hanya jika kunci belum ada di metrik akhir

                    # optional: log keys for debug
                    self.logger.info(f"Reporter metrics keys being passed: {list(final_metrics.keys())}") # Catat kunci metrik yang akan dikirim ke reporter

                    # main.py sebelum self.reporter.run()
                    print("=== COLUMNS df_final ===") 
                    print(df_final.columns) 
                    print("=== LAST ROW df_final ===")
                    print(df_final.tail(1))

                    # 7) run reporter
                    self.reporter.run(df_final, final_metrics, anomalies) # Panggil fungsi pelaporan dengan DataFrame hasil akhir, metrik akhir, dan anomali yang terdeteksi



            # NEW: Realtime inference pipeline (opsional, tergantung config)
            if hasattr(self.config.REALTIME, "enable_realtime_inference") and \
               self.config.REALTIME.enable_realtime_inference:
                self.run_realtime_inference() # Panggil fungsi untuk menjalankan inferensi realtime

            if self.config.REALTIME.enable_monitoring: # Cek apakah konfigurasi mengizinkan pemantauan realtime
                self.start_monitoring_loop() # Panggil fungsi untuk memulai loop pemantauan realtime

        except Exception as e: # Tangkap error jika terjadi masalah selama eksekusi
            self.logger.critical(f"PIPELINE FAILED: {e}", exc_info=True) # Catat error kritis ke log dengan informasi traceback
        finally: # Blok finally untuk eksekusi kode pembersihan
            end_ts = time.time() # Catat waktu selesai eksekusi
            self.logger.info(f"System shutdown. Uptime: {end_ts - start_ts:.2f}s") # Catat info waktu eksekusi ke log


# ==============================================================================
# CLI ARGUMENT SETUP
# ==============================================================================

def create_arg_parser() -> argparse.ArgumentParser: # Fungsi untuk membuat parser argumen CLI
    parser = argparse.ArgumentParser(description="VolcanoAI Titan System") # Inisialisasi parser argumen dengan deskripsi
    parser.add_argument("--skip-training", action="store_true") # Argumen untuk melewati pelatihan model
    parser.add_argument("--eval-only", action="store_true") # Argumen untuk hanya menjalankan evaluasi model
    parser.add_argument("--no-ga", action="store_true") # Argumen untuk menonaktifkan engine GA
    parser.add_argument("--no-aco", action="store_true") # Argumen untuk menonaktifkan engine ACO
    parser.add_argument("--no-reporting", action="store_true") # Argumen untuk menonaktifkan pelaporan
    return parser # Kembalikan parser argumen yang sudah dibuat


def configure_pipeline_from_args(args: argparse.Namespace, config: ProjectConfig): # Fungsi untuk mengonfigurasi pipeline berdasarkan argumen CLI
    if args.skip_training: # Jika argumen --skip-training diberikan
        config.PIPELINE.run_model_training = False # Nonaktifkan pelatihan model
    if args.eval_only: # Jika argumen --eval-only diberikan
        config.PIPELINE.run_model_training = False # Nonaktifkan pelatihan model
        config.PIPELINE.run_data_loading = False # Nonaktifkan pemuatan data
        config.PIPELINE.run_feature_engineering = False # Nonaktifkan rekayasa fitur
    if args.no_ga: # Jika argumen --no-ga diberikan
        config.PIPELINE.run_ga_engine = False # Nonaktifkan engine GA
    if args.no_aco: # Jika argumen --no-aco diberikan
        config.PIPELINE.run_aco_engine = False # Nonaktifkan engine ACO
    if args.no_reporting: # Jika argumen --no-reporting diberikan
        config.PIPELINE.run_reporting = False # Nonaktifkan pelaporan


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main(): # Fungsi utama sebagai titik masuk program
    parser = create_arg_parser() # Buat parser argumen CLI
    args = parser.parse_args() # Parse argumen dari command line
     
    if not SYSTEM_READY: # Cek apakah sistem siap dijalankan
        print("System tidak siap. Ada modul yang hilang.") # Tampilkan pesan error
        return

    os.makedirs(CONFIG.OUTPUT.directory, exist_ok=True) # Buat direktori output jika belum ada
    setup_logging(CONFIG.OUTPUT.directory) # Atur logging ke direktori output

    logging.info("=" * 80) 
    logging.info("  VOLCANOAI SYSTEM STARTUP  ".center(80))
    logging.info("=" * 80)

    configure_pipeline_from_args(args, CONFIG) # Konfigurasikan pipeline berdasarkan argumen CLI

    pipeline = VolcanoAiPipeline(CONFIG) # Inisialisasi instance pipeline dengan konfigurasi proyek
    pipeline.run() # Jalankan pipeline

# === START: Flask dashboard integration (paste di akhir main.py) ===
import os
import time
import json
import logging
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
import pandas as pd
from flask import Flask, send_from_directory, Response
from jinja2 import Template

# ===============================
# CONFIG DEFAULTS
# ===============================
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
TEMPLATE_PATH = Path("VolcanoAI/reporting/templates/monitor_live_template.html")
PROJECT_ROOT = Path.cwd()

# ===============================
# UTILITY FUNCTIONS
# ===============================
def _file_url_for(path: Path) -> str:
    """Convert file-system path to URL served by /files/... route."""
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except Exception:
        rel = path
    return f"/files/{rel.as_posix()}"

def get_latest_confusion_matrix(nb_dir: Path) -> Path | None:
    """
    Ambil path confusion matrix terbaru sesuai mekanisme NB engine.
    Priority:
    1) confusion_matrix_latest.txt (paling akurat)
    2) confusion_matrix_*.png (fallback)
    """
    if not nb_dir.exists():
        return None

    pointer = nb_dir / "confusion_matrix_latest.txt"
    if pointer.exists():
        try:
            fname = pointer.read_text(encoding="utf-8").strip()
            p = nb_dir / fname
            if p.exists():
                return p
        except Exception:
            pass

    # fallback: glob timestamped
    files = sorted(
        nb_dir.glob("confusion_matrix_*.png"),
        reverse=True
    )
    return files[0] if files else None

def build_dashboard_context(output_dir: Path) -> dict:
    """Baca semua file output untuk dashboard, isi context dict."""
    ctx = {
        "TIMESTAMP": datetime.now().isoformat(sep=' '),
        "NOW_TS": int(time.time()),
        "ACO_IMPACT_CENTER": "N/A",
        "ACO_IMPACT_AREA": "N/A",
        "ACO_MAP": "#",
        "GA_MAP": "#",
        "GA_PRED_LAT": "N/A",
        "GA_PRED_LON": "N/A",
        "GA_BEARING": "N/A",
        "GA_DISTANCE": "N/A",
        "GA_CONFIDENCE": "N/A",
        "LATEST_ROW_HTML": "<em>No data yet</em>",
        "LSTM_MASTER_CSV": "#",
        "LSTM_MASTER_FILENAME": "N/A",
        "LSTM_RECENT_CSV": "#",
        "LSTM_RECENT_FILENAME": "N/A",
        "LSTM_ANOMALIES_CSV": "#",
        "LSTM_ANOMALIES_FILENAME": "N/A",
        "CNN_PRED_CSV": "#",
        "CNN_PRED_JSON": "#",
        "CNN_IMAGE_LIST_HTML": "",
        "NB_METRICS_HTML": "<em>Not available</em>",
        "NB_CONFUSION_PNG": "#",
        "NB_ROC_PNG": "#",
        "NB_REPORT_STR": "N/A"
    }

    out = output_dir

    # ===============================
    # ACO RESULTS
    # ===============================
    aco_json = out / "aco_results" / "aco_to_ga.json"
    if aco_json.exists():
        try:
            j = json.loads(aco_json.read_text(encoding="utf-8"))
            lat = j.get("center_lat")
            lon = j.get("center_lon")
            if lat is not None and lon is not None:
                ctx["ACO_IMPACT_CENTER"] = f"{lat}, {lon}"
            ctx["ACO_IMPACT_AREA"] = j.get("impact_area_km2", ctx["ACO_IMPACT_AREA"])
            aco_map_file = out / "aco_results" / "aco_impact_zones.html"
            if aco_map_file.exists():
                ctx["ACO_MAP"] = _file_url_for(aco_map_file)
        except Exception:
            pass

    # ===============================
    # GA RESULTS
    # ===============================
    ga_map = out / "ga_results" / "ga_from_aco_map.html"
    if ga_map.exists():
        ctx["GA_MAP"] = _file_url_for(ga_map)

    try:
        if aco_json.exists():
            j = json.loads(aco_json.read_text(encoding="utf-8"))
            ga_path = j.get("ga_path", [])
            if isinstance(ga_path, list) and len(ga_path) > 0:
                first = ga_path[0]
                angle = first.get("angle_deg")
                direction = first.get("direction")
                distance = first.get("distance_km")
                if angle is not None:
                    ctx["GA_BEARING"] = f"{angle}° ({direction})" if direction else f"{angle}°"
                if distance is not None:
                    ctx["GA_DISTANCE"] = f"{distance:.2f} km"
        ga_vec = out / "ga_results" / "ga_vector.json"
        if ga_vec.exists():
            gv = json.loads(ga_vec.read_text(encoding="utf-8"))
            if "distance_km" in gv:
                ctx["GA_DISTANCE"] = f"{gv['distance_km']:.2f} km"
            if ctx["GA_BEARING"] == "N/A" and "bearing_degree" in gv:
                ctx["GA_BEARING"] = f"{gv['bearing_degree']}°"
    except Exception:
        pass

    # ===============================
    # LSTM RESULTS
    # ===============================
    lstm_dir = out / "lstm_results"
    if lstm_dir.exists():
        for f in ["lstm_records_2y_20241230.csv", "master.csv"]:
            p = lstm_dir / f
            if p.exists():
                ctx["LSTM_MASTER_CSV"] = _file_url_for(p)
                ctx["LSTM_MASTER_FILENAME"] = p.name
                break
        for f in ["lstm_recent_15d_20241230.csv", "recent.csv"]:
            p = lstm_dir / f
            if p.exists():
                ctx["LSTM_RECENT_CSV"] = _file_url_for(p)
                ctx["LSTM_RECENT_FILENAME"] = p.name
                break
        for f in ["lstm_anomalies_20241230.csv", "anomalies.csv"]:
            p = lstm_dir / f
            if p.exists():
                ctx["LSTM_ANOMALIES_CSV"] = _file_url_for(p)
                ctx["LSTM_ANOMALIES_FILENAME"] = p.name
                break

    # ===============================
    # CNN RESULTS
    # ===============================
    cnn_latest = out / "cnn_results" / "results" / "cnn_predictions_latest.csv"
    if cnn_latest.exists():
        ctx["CNN_PRED_CSV"] = _file_url_for(cnn_latest)
        try:
            df_cnn = pd.read_csv(cnn_latest, parse_dates=["Acquired_Date"])
            if not df_cnn.empty:
                # [FIX DASHBOARD]: Paksa urutkan data berdasarkan Tanggal, 
                # sehingga baris data Tahun 2026 PASTI berada di paling bawah!
                df_cnn = df_cnn.sort_values(by="Acquired_Date", ascending=True)
                
                # Ambil 1 baris terbawah yang merupakan event paling akhir/terbaru
                last_event = df_cnn.tail(1).copy()
                
                # Ubah isi yang kosong (NaN) agar tabel HTML tidak error
                last_event = last_event.fillna("-")
                
                ctx["LATEST_ROW_HTML"] = last_event.to_html(index=False, classes="table", border=0)
        except Exception as e:
            ctx["LATEST_ROW_HTML"] = f"<em>Error memproses data: {e}</em>"
            pass

    cnn_json = out / "cnn_results" / "cnn_predictions_latest.json"
    if cnn_json.exists():
        ctx["CNN_PRED_JSON"] = _file_url_for(cnn_json)

    cnn_img = out / "cnn_results" / "cnn_prediction_map.png"
    html_imgs = []
    if cnn_img.exists():
        html_imgs.append(f'<img src="{_file_url_for(cnn_img)}" class="plot" alt="CNN Prediction Map">')

    if html_imgs:
        ctx["CNN_IMAGE_LIST_HTML"] = "<br><br>".join(html_imgs)
    else:
        ctx["CNN_IMAGE_LIST_HTML"] = "<p class='muted'></p>"

    pointer = out / "cnn_results" / "maps" / "latest_map.txt"
    if pointer.exists():
        try:
            p = Path(pointer.read_text(encoding="utf-8").strip())
            if p.exists():
                ctx["CNN_MAP"] = _file_url_for(p)
            else:
                ctx["CNN_MAP"] = "#"
        except Exception:
            ctx["CNN_MAP"] = "#"
    else:
        ctx["CNN_MAP"] = "#"

    # ===============================
    # NAIVE BAYES RESULTS
    # ===============================
    nb_candidates = [
        out / "naive_bayes_results",
        out / "naive_bayes",
        out / "naive_bayes_outputs",
    ]

    nb_found = None
    for cand in nb_candidates:
        if cand.exists():
            nb_found = cand
            break
    if nb_found:
        latest_txt = nb_found / "confusion_matrix_latest.txt"
        if latest_txt.exists():
            try:
                fname = latest_txt.read_text(encoding="utf-8").strip()
                p1 = nb_found / fname
            except Exception:
                p1 = nb_found / "confusion_matrix.png"
        else:
            p1 = nb_found / "confusion_matrix.png"
        p2 = nb_found / "roc_curves.png"
        p_csv = nb_found / "naive_bayes_predictions.csv"
        p_json = nb_found / "naive_bayes_metrics.json"
        p_report = nb_found / "classification_report.txt"

        if p1.exists():
            ctx["NB_CONFUSION_PNG"] = f"{_file_url_for(p1)}?v={ctx['NOW_TS']}"
        if p2.exists():
            ctx["NB_ROC_PNG"] = f"{_file_url_for(p2)}?v={ctx['NOW_TS']}"
        if p_csv.exists():
            ctx["NB_PRED_CSV"] = _file_url_for(p_csv)

        if p_json.exists():
            try:
                mj = json.loads(p_json.read_text(encoding="utf-8"))
                rows = ["<table>"]
                for k, v in mj.items():
                    if k == "roc_auc":
                        continue
                    rows.append(f"<tr><th style='text-align:left;padding:6px'>{k}</th><td style='padding:6px'>{v}</td></tr>")
                rows.append("</table>")
                ctx["NB_METRICS_HTML"] = "\n".join(rows)
            except Exception:
                ctx["NB_METRICS_HTML"] = "<em>Metrics JSON exists but failed to parse</em>"

        if p_report.exists():
            try:
                ctx["NB_REPORT_STR"] = p_report.read_text(encoding="utf-8")
            except Exception:
                ctx["NB_REPORT_STR"] = "<em>Failed to read classification report</em>"

    return ctx

# ===============================
# FLASK APP
# ===============================
def create_flask_app(output_dir: Path, template_path: Path) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        ctx = build_dashboard_context(output_dir)
        if template_path.exists():
            tpl_text = template_path.read_text(encoding="utf-8")
            tmpl = Template(tpl_text)
            rendered = tmpl.render(**ctx)
            return Response(rendered, mimetype="text/html")
        else:
            return "<h3>Template not found</h3><p>Check TEMPLATE_PATH setting.</p>", 404

    @app.route("/files/<path:filename>")
    def serve_file(filename):
        full = (PROJECT_ROOT / Path(filename)).resolve()
        try:
            full.relative_to(PROJECT_ROOT.resolve())
        except Exception:
            return "Forbidden", 403
        if not full.exists():
            return "Not found", 404
        resp = send_from_directory(PROJECT_ROOT, filename, conditional=True)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    return app

def start_flask_in_thread(app: Flask, host="127.0.0.1", port=5000, open_browser=True):
    def _run():
        if open_browser:
            try:
                webbrowser.open_new_tab(f"http://{host}:{port}")
            except Exception:
                pass
        app.run(host=host, port=port, debug=False, use_reloader=False)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th

# ===============================
# MAIN WITH DASHBOARD
# ===============================
def main_with_dashboard():
    parser = create_arg_parser()
    args = parser.parse_args()

    if not SYSTEM_READY:
        print("System tidak siap. Ada modul yang hilang.")
        return

    os.makedirs(CONFIG.OUTPUT.directory, exist_ok=True)
    setup_logging(CONFIG.OUTPUT.directory)
    out = Path(CONFIG.OUTPUT.directory)
    out.mkdir(parents=True, exist_ok=True)

    logging.info("="*80)
    logging.info("  VOLCANOAI SYSTEM STARTUP  ".center(80))
    logging.info("="*80)

    configure_pipeline_from_args(args, CONFIG)

    # CREATE FLASK APP
    app = create_flask_app(output_dir=out, template_path=TEMPLATE_PATH)

    enable_monitoring = getattr(CONFIG.REALTIME, "enable_monitoring", False)
    if enable_monitoring:
        logging.info("[Dashboard] Starting Flask server in background thread (monitoring enabled)...")
        start_flask_in_thread(app, host=FLASK_HOST, port=FLASK_PORT, open_browser=True)

    pipeline = VolcanoAiPipeline(CONFIG)
    pipeline.run()

    if not enable_monitoring:
        logging.info("[Dashboard] Pipeline finished — launching Flask server (dashboard)...")
        start_flask_in_thread(app, host=FLASK_HOST, port=FLASK_PORT, open_browser=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("[Dashboard] KeyboardInterrupt — exiting.")

# Run as script
if __name__ == "__main__":
    main_with_dashboard()
# === END: Flask dashboard integration ===

# VolcanoAI/engines/naive_bayes_engine.py  (repaired & CNN-Integrated)
# -- coding: utf-8 --
"""
Naive Bayes Engine (Terintegrasi Output CNN):
- Target ``Normal`` / ``Tidak Normal`` kini dievaluasi berdasarkan Error output CNN 
  (Arah dan Sudut prediksi CNN vs Kejadian Aktual)
- Membersihkan decimal koma pada kolom koordinat
- Menyediakan mapping coords sederhana untuk beberapa gunung contoh
- Menghindari "label leakage": target hanya bergantung pada validation prediksi spasial
"""

import os
import logging
import pickle
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import shutil
import math

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

logger = logging.getLogger("VolcanoAI.NaiveBayesEngine")
logger.addHandler(logging.NullHandler())


# ------------------ Helpers ------------------

def _convert(obj):
    import numpy as _np
    from datetime import datetime as _dt
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, _dt):
        return obj.isoformat()
    return str(obj)


def _atomic_write(path, data_bytes):
    dirn = os.path.dirname(path)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=dirn)
    try:
        tmp.write(data_bytes)
        tmp.flush()
        tmp.close()
        os.replace(tmp.name, path)
    finally:
        if os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except:
                pass


def haversine_km(lat1, lon1, lat2, lon2):
    """Return distance in km between pairs (arrays/scalars)."""
    try:
        lat1 = np.asarray(lat1, dtype=float)
        lon1 = np.asarray(lon1, dtype=float)
        lat2 = np.asarray(lat2, dtype=float)
        lon2 = np.asarray(lon2, dtype=float)
    except Exception:
        return float("inf")

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    a = np.minimum(1.0, np.maximum(0.0, a))
    R = 6371.0
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


# ------------------ Preprocessor ------------------
class ClassificationPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.features_in: List[str] = self.config.get("features", [])
        self.k_best = int(self.config.get("k_best_features", 7))
        self.pipeline: Optional[Pipeline] = None
        self.selected_features: List[str] = []
        self.feature_columns_: List[str] = [] 

    def fit(self, df: pd.DataFrame, target: Any):
        valid_features = [f for f in self.features_in if f in df.columns]
        if not valid_features:
            valid_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if not valid_features:
                raise ValueError("Tidak ada fitur numeric untuk training.")

        self.feature_columns_ = list(valid_features)
        X = df[self.feature_columns_].copy()
        y = pd.Series(target) if not isinstance(target, pd.Series) else target

        k = max(1, min(self.k_best, X.shape[1]))
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", SkStandardScaler()),
            ("selector", SelectKBest(f_classif, k=k)),
        ]

        try:
            if y.nunique() <= 1:
                self.pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", SkStandardScaler())])
                self.pipeline.fit(X)
                self.selected_features = self.feature_columns_
                return
        except Exception:
            pass

        self.pipeline = Pipeline(steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipeline.fit(X, y)

        if "selector" in self.pipeline.named_steps:
            mask = self.pipeline.named_steps["selector"].get_support()
            self.selected_features = [f for f, m in zip(self.feature_columns_, mask) if m]
        else:
            self.selected_features = self.feature_columns_

    def transform(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.pipeline is None:
            return None
        X = pd.DataFrame(index=df.index)
        for col in self.feature_columns_:
            if col in df.columns:
                X[col] = df[col]
            else:
                X[col] = np.nan
        try:
            return self.pipeline.transform(X)
        except Exception as e:
            logger.warning(f"Transform failed: {e}. Returning zeros.")
            return np.zeros((len(df), max(1, len(self.feature_columns_))))

    def fit_transform(self, df: pd.DataFrame, target: Any) -> np.ndarray:
        self.fit(df, target)
        X = pd.DataFrame(index=df.index)
        for col in self.feature_columns_:
            X[col] = df[col] if col in df.columns else np.nan
        return self.pipeline.transform(X)


# ------------------ Evaluator & Reporter ------------------
class ModelEvaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def evaluate(self, y_true, y_pred, y_prob: Optional[np.ndarray] = None, inputs_are_encoded: bool = False) -> Dict[str, Any]:
        metrics = {"class_names": self.class_names}
        try:
            if inputs_are_encoded:
                y_true_arr = np.asarray(y_true).astype(int)
                y_pred_arr = np.asarray(y_pred).astype(int)
                labels = list(range(len(self.class_names)))
                target_names = self.class_names
            else:
                y_true_arr = np.asarray(y_true).astype(object)
                y_pred_arr = np.asarray(y_pred).astype(object)
                labels = self.class_names
                target_names = self.class_names

            metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr)) #Rumus 3.50
            metrics["confusion_matrix"] = confusion_matrix(y_true_arr, y_pred_arr, labels=labels) #Rumus 3.51, 3.52, 3.53
            metrics["report_str"] = classification_report(y_true_arr, y_pred_arr, labels=labels, target_names=target_names, zero_division=0)

            if y_prob is not None and isinstance(y_prob, np.ndarray):
                if not inputs_are_encoded:
                    y_true_enc = self._str_to_encoded(y_true)
                else:
                    y_true_enc = np.asarray(y_true).astype(int)
                metrics["roc_auc"] = self._calculate_roc_auc(y_true_enc, y_prob)

        except Exception as e:
            logger.error(f"Evaluasi gagal: {e}", exc_info=True)

        return metrics

    def _str_to_encoded(self, y_strs):
        mapping = {label: idx for idx, label in enumerate(self.class_names)}
        return np.array([mapping.get(v, -1) for v in y_strs], dtype=int)

    def _calculate_roc_auc(self, y_true, y_prob) -> Dict[str, Any]:
        n_classes = len(self.class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            if i >= y_prob.shape[1]:
                continue
            if np.sum(y_bin[:, i]) == 0:
                roc_auc[self.class_names[i]] = 0.5
                continue
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc[self.class_names[i]] = auc(fpr[i], tpr[i])
        return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}


class ClassificationReporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_plots(self, metrics: Dict[str, Any]):
        if "confusion_matrix" in metrics:
            self._plot_cm(metrics["confusion_matrix"], metrics["class_names"])
        if "roc_auc" in metrics and "fpr" in metrics["roc_auc"]:
            self._plot_roc(metrics["roc_auc"])

    def _plot_cm(self, cm, classes):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"confusion_matrix_{ts}.png"

        fullpath = os.path.join(self.output_dir, fname)
        plt.savefig(fullpath)

        static_dir = os.path.join("static", "naive_bayes")
        os.makedirs(static_dir, exist_ok=True)
        static_path = os.path.join(static_dir, "confusion_matrix_latest.png")
        shutil.copyfile(fullpath, static_path)

        plt.close()

    def _plot_roc(self, roc_data):
        plt.figure(figsize=(8, 6))
        for cls, area in roc_data["roc_auc"].items():
            if cls in roc_data["fpr"]:
                plt.plot(roc_data["fpr"][cls], roc_data["tpr"][cls], label=f'{cls} (AUC = {area:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve per Level')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"))
        plt.close()


# ------------------ NaiveBayesEngine (binary) ------------------
class NaiveBayesEngine:
    def __init__(self, config: Any):
        self.config = config.__dict__ if not isinstance(config, dict) else config
        self.output_dir = self.config.get("output_dir", "output/naive_bayes_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.class_names = ["Normal", "Tidak Normal"]
        self.target_col = "binary_status"

        # --- CNN vs AKTUAL THRESHOLDS (New Logic) ---
        self.MAX_ANGLE_ERR = float(self.config.get("nb_max_angle_err", 45.0)) # Derajat
        self.MAX_DIST_ERR = float(self.config.get("nb_max_dist_err", 15.0))   # Km

        # Fallback thresholds
        self.DIST_THRESHOLD_KM = float(self.config.get("nb_dist_threshold_km", 90.0))
        self.MAG_THRESHOLD = float(self.config.get("nb_mag_threshold", 4.5))

        self.volcano_coords = {
            "Kelud": (-7.93, 112.27),
            "Raung": (-8.12, 114.03),
            "Bromo": (-7.9425, 112.9530),
            "Semeru": (-8.1083, 112.9229),
        }

        self.model_path = os.path.join(self.output_dir, "naive_bayes_model.pkl")
        self.preproc_path = os.path.join(self.output_dir, "preprocessor.pkl")
        self.le_path = os.path.join(self.output_dir, "label_encoder.pkl")

        self.le = LabelEncoder()
        self.le.fit(self.class_names)

        self.model: Optional[GaussianNB] = None
        self.preprocessor = ClassificationPreprocessor(self.config)
        self.evaluator = ModelEvaluator(self.class_names)
        self.reporter = ClassificationReporter(self.output_dir)

    def _clean_numeric_commas(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in ["Lintang", "Bujur", "EQ_Lintang", "EQ_Bujur", "Magnitudo", "Kedalaman (km)"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def _extract_volcano_name(self, df: pd.DataFrame, source_col: str = "Lokasi") -> pd.Series:
        if source_col not in df.columns:
            return pd.Series(['Unknown'] * len(df), index=df.index)
        names = (
            df[source_col].astype(str)
              .str.replace('Gunung', '', regex=False)
              .str.split(',').str[0].str.strip().str.title()
        )
        return names

    def _compute_distance_to_volcano(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df = self._clean_numeric_commas(df)
        vol_names = self._extract_volcano_name(df, source_col='Lokasi' if 'Lokasi' in df.columns else 'Nama')
        dist = np.full(len(df), np.nan)

        if 'EQ_Lintang' in df.columns and 'EQ_Bujur' in df.columns:
            for idx in range(len(df)):
                vn = vol_names.iloc[idx]
                try:
                    lat_e = float(df.iloc[idx]['EQ_Lintang'])
                    lon_e = float(df.iloc[idx]['EQ_Bujur'])
                except Exception:
                    dist[idx] = np.nan
                    continue
                if vn in self.volcano_coords:
                    lat_v, lon_v = self.volcano_coords[vn]
                    dist[idx] = float(haversine_km(lat_e, lon_e, lat_v, lon_v))
                else:
                    dist[idx] = np.nan
        else:
            dist[:] = np.nan

        return pd.Series(dist, index=df.index)

    # --- CNN ERROR EVALUATION ---
    def _calculate_cnn_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung Error CNN (Prediksi vs Kejadian Aktual)"""
        df = df.copy()
        # Hitung Error Sudut (Mempertimbangkan derajat sirkular, max 180 diff)
        if 'cnn_angle_deg' in df.columns and 'ga_bearing_deg' in df.columns:
            angle_err = (df['cnn_angle_deg'] - df['ga_bearing_deg']).abs() % 360
            df['cnn_angle_err'] = angle_err.apply(lambda x: 360 - x if x > 180 else x)
        
        # Hitung Error Jarak
        if 'cnn_distance_km' in df.columns and 'ga_distance_km' in df.columns:
            df['cnn_dist_err'] = (df['cnn_distance_km'] - df['ga_distance_km']).abs()
            
        return df

    # --- TARGET GENERATION (NEW LOGIC) ---
    def _calculate_binary_status(self, df: pd.DataFrame) -> pd.Series: #input naive
        """
        Labeling BERDASARKAN OUTPUT CNN vs DATA AKTUAL.
        Jika error prediksi (arah/jarak) melebihi threshold -> "Tidak Normal" (Anomali aktivitas)
        Jika sesuai dengan prediksi CNN -> "Normal" (Sesuai tren spasial)
        """
        df = df.copy()
        
        has_cnn_error_cols = 'cnn_angle_err' in df.columns and 'cnn_dist_err' in df.columns
        
        if has_cnn_error_cols:
            logger.info("[NB] Menggunakan output prediksi CNN vs Aktual untuk menentukan target kelas.")
            
            # Jika salah satu dari error Arah ATAU Jarak melampaui toleransi, maka terhitung "Tidak Normal"
            mask_anomaly = (df['cnn_angle_err'] > self.MAX_ANGLE_ERR) | (df['cnn_dist_err'] > self.MAX_DIST_ERR)
            status = np.where(mask_anomaly, 'Tidak Normal', 'Normal')
            return pd.Series(status, index=df.index)
            
        else:
            # FALLBACK Jika pipeline CNN belum jalan
            logger.warning("[NB] Output Error CNN tidak ditemukan! Fallback ke penentuan target berbasis Magnitudo.")
            df = self._clean_numeric_commas(df)
            if 'distance_km' not in df.columns:
                df['distance_km'] = self._compute_distance_to_volcano(df)

            mag = df['Magnitudo'] if 'Magnitudo' in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
            dist = df['distance_km'].fillna(1e9)

            mask = (dist <= self.DIST_THRESHOLD_KM) & (mag.fillna(0.0) >= self.MAG_THRESHOLD)
            status = np.where(mask, 'Tidak Normal', 'Normal')
            return pd.Series(status, index=df.index)

    # --- train / evaluate ---
    def train(self, df_train: pd.DataFrame) -> bool:
        df_train = df_train.copy()

        # Kalkulasi error CNN jika datanya sudah ada
        df_train = self._calculate_cnn_errors(df_train)

        for col in self.preprocessor.features_in:
            if col not in df_train.columns:
                df_train[col] = 0.0
                logger.warning(f"[NB] Feature '{col}' missing → filled with 0")

        # create binary target dari error CNN
        df_train[self.target_col] = self._calculate_binary_status(df_train)

        logger.info(f"Distribusi Kelas Training (binary):\n{df_train[self.target_col].value_counts().to_dict()}")

        y_raw = df_train[self.target_col].astype(str)
        try:
            y_encoded = self.le.transform(y_raw)
        except Exception as e:
            logger.warning(f"Label mismatch: {e}. Reset encoder to binary classes.")
            self.le.fit(self.class_names)
            y_encoded = self.le.transform(y_raw)

        X_processed = self.preprocessor.fit_transform(df_train, y_encoded)
        if X_processed is None:
            logger.error("Preprocessor returned None during training.")
            return False

        self.model = GaussianNB()
        self.model.fit(X_processed, y_encoded)

        self._save_artifacts()
        logger.info("Training Naive Bayes (binary) selesai.")
        return True

    def evaluate(self, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.model is None:
            if not self._load_artifacts():
                logger.error("Model NB belum siap.")
                return df_test, {}

        df_out = df_test.copy()
        
        # Ekstrak Error CNN vs Aktual dulu sebelum dilabeli
        df_out = self._calculate_cnn_errors(df_out)
        df_out[self.target_col] = self._calculate_binary_status(df_out)

        if hasattr(self.preprocessor, "feature_columns_") and self.preprocessor.feature_columns_:
            for col in self.preprocessor.feature_columns_:
                if col not in df_out.columns:
                    df_out[col] = np.nan

        X_processed = self.preprocessor.transform(df_out)
        if X_processed is None:
            logger.error("Preprocessor transform returned None. Aborting eval.")
            return df_out, {}

        preds = self.model.predict(X_processed)
        try:
            probs_raw = self.model.predict_proba(X_processed)
        except Exception:
            probs_raw = np.zeros((len(df_out), len(self.class_names)))
            for i, p in enumerate(preds):
                probs_raw[i, int(p)] = 1.0

        probs_full = np.zeros((len(df_out), len(self.class_names)))
        for model_idx, class_label in enumerate(self.model.classes_):
            if model_idx < probs_raw.shape[1]:
                probs_full[:, int(class_label)] = probs_raw[:, model_idx]

        try:
            pred_labels = self.le.inverse_transform(preds)
        except Exception:
            pred_labels = np.array([self.class_names[int(p)] if int(p) < len(self.class_names) else "Unknown" for p in preds])

        df_out["kelas_prediksi"] = pred_labels
        df_out["nb_confidence"] = np.max(probs_full, axis=1)
        df_out["anomaly_score"] = 1.0 - df_out["nb_confidence"]
        df_out["is_anomaly"] = df_out["anomaly_score"] > 0.5

        logger.info("Unique true labels (raw): %s", df_out[self.target_col].unique())
        logger.info("Unique pred labels (str): %s", df_out["kelas_prediksi"].unique())

        metrics = {}
        try:
            metrics = self.evaluator.evaluate(
                df_out[self.target_col].astype(str).tolist(),
                df_out["kelas_prediksi"].astype(str).tolist(),
                probs_full,
                inputs_are_encoded=False
            )
            self.reporter.generate_plots(metrics)
        except Exception as e:
            logger.warning(f"Evaluasi skip (data mungkin kurang / error): {e}", exc_info=True)

        self._save_outputs(df_out, metrics)
        return df_out, metrics

    def _save_outputs(self, df_out, metrics):
        pred_path = os.path.join(self.output_dir, "naive_bayes_predictions.csv")
        tmp_csv = pred_path + f".{int(time.time())}.tmp"
        df_out.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, pred_path)

        metrics_path = os.path.join(self.output_dir, "naive_bayes_metrics.json")
        metrics_ts_path = os.path.join(self.output_dir, f"naive_bayes_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        metrics_bytes = json.dumps(metrics, indent=2, default=_convert).encode("utf-8")
        _atomic_write(metrics_ts_path, metrics_bytes)
        _atomic_write(metrics_path, metrics_bytes)

        if "report_str" in metrics:
            report_path = os.path.join(self.output_dir, "classification_report.txt")
            _atomic_write(report_path, metrics["report_str"].encode("utf-8"))

        with open(os.path.join(self.output_dir, "latest_metrics.json"), "w", encoding="utf-8") as f:
            f.write(os.path.basename(metrics_path))

        try:
            self._export_presentation_excel(df_out, metrics)
        except Exception as e:
            logger.warning(f"[NB] Export presentation Excel failed: {e}", exc_info=True)


    def _export_presentation_excel(self, df_out: pd.DataFrame, metrics: Dict[str, Any], filename: Optional[str] = None) -> Optional[str]:
        try:
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fname = filename if filename else f"naive_bayes_presentation_{ts}.xlsx"
            out_path = out_dir / fname

            keep_cols = []
            # Menambahkan indikator Error CNN agar mempermudah validasi Dosen Penguji
            pref = [
                "Acquired_Date", "Nama", "Lokasi", #output naive
                "EQ_Lintang", "EQ_Bujur", "Magnitudo",
                "cnn_angle_deg", "ga_bearing_deg", "cnn_angle_err",
                "cnn_distance_km", "ga_distance_km", "cnn_dist_err",
                self.target_col, "kelas_prediksi", "nb_confidence", "is_anomaly"
            ]
            for c in pref:
                if c in df_out.columns:
                    keep_cols.append(c)

            if not keep_cols:
                summary_df = df_out.copy()
            else:
                summary_df = df_out[keep_cols].copy()

            if 'Acquired_Date' in summary_df.columns:
                summary_df['Acquired_Date'] = pd.to_datetime(summary_df['Acquired_Date'], errors='coerce').astype(str)

            normal_df = summary_df[summary_df[self.target_col] == "Normal"] if self.target_col in summary_df.columns else pd.DataFrame()
            tidaknormal_df = summary_df[summary_df[self.target_col] == "Tidak Normal"] if self.target_col in summary_df.columns else pd.DataFrame()

            meta = {
                "generated_at": datetime.utcnow().isoformat(),
                "engine": "NaiveBayesEngine (CNN Target Evaluator)",
                "cnn_max_angle_err_threshold": self.MAX_ANGLE_ERR,
                "cnn_max_dist_err_threshold": self.MAX_DIST_ERR,
                "rows_input": int(len(df_out)),
                "notes": "Target kelas dievaluasi berdasar seberapa akurat prediksi CNN terhadap realitas spasial."
            }
            try:
                meta["accuracy"] = metrics.get("accuracy", None)
            except Exception:
                pass

            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="NB_Summary", index=False)
                if not normal_df.empty:
                    normal_df.to_excel(writer, sheet_name="NB_Normal", index=False)
                if not tidaknormal_df.empty:
                    tidaknormal_df.to_excel(writer, sheet_name="NB_TidakNormal", index=False)
                pd.DataFrame([meta]).to_excel(writer, sheet_name="Meta", index=False)
                try:
                    df_out.to_excel(writer, sheet_name="Raw_Predictions", index=False)
                except Exception:
                    pd.DataFrame(df_out).to_excel(writer, sheet_name="Raw_Predictions", index=False)

            logger.info(f"[NB] Presentation Excel saved → {out_path}")
            return str(out_path)
        except Exception as e:
            logger.error(f"[NB] _export_presentation_excel failed: {e}", exc_info=True)
            return None

    def _save_artifacts(self):
        try:
            with open(self.model_path, "wb") as f: pickle.dump(self.model, f)
            with open(self.preproc_path, "wb") as f: pickle.dump(self.preprocessor, f)
            with open(self.le_path, "wb") as f: pickle.dump(self.le, f)
        except Exception as e:
            logger.error(f"Gagal simpan artifacts: {e}")

    def _load_artifacts(self) -> bool:
        try:
            with open(self.model_path, "rb") as f: self.model = pickle.load(f)
            with open(self.preproc_path, "rb") as f: self.preprocessor = pickle.load(f)
            with open(self.le_path, "rb") as f: self.le = pickle.load(f)
            return True
        except Exception:
            return False
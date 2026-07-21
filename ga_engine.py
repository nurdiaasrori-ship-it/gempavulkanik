# ============================================
# VolcanoAI/engines/ga_engine.py
# GA Engine + Vector Prediction + Map Popup
# ============================================

import os # operating system
import sys # system
import json # JSON handling
import time # time measurement
import math # mathematical functions
import random # random number generation
import shutil # file operations
import pickle # object serialization
import functools # function tools
import logging # logging
import warnings # warnings management
import uuid # unique identifiers
from datetime import datetime # date and time handling
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterable # type hints

import numpy as np # numerical computing
import pandas as pd # data manipulation
import networkx as nx # graph algorithms
import folium # interactive maps
from folium import plugins # folium plugins
from folium.plugins import AntPath, HeatMap, Fullscreen, MiniMap, MeasureControl # folium extras

from deap import base, creator, tools, algorithms # DEAP evolutionary algorithms

import matplotlib # plotting
matplotlib.use("Agg") # non-GUI backend
import matplotlib.pyplot as plt # plotting
import seaborn as sns # statistical data visualization
from scipy.spatial.distance import pdist, squareform # distance computations


# =======
# Logging
# =======
logger = logging.getLogger("VolcanoAI.GaEngine")
logger.addHandler(logging.NullHandler())


# ==========================================
# FIX: Prevent DEAP Creator Crash (REQUIRED)
# ==========================================
if not hasattr(creator, "FitnessMin"): 
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


# ===========================
# Utility Decorators
# ===========================
def execution_monitor(func): # decorator untuk monitor eksekusi fungsi
    @functools.wraps(func)
    def wrapper(*args, **kwargs): # wrapper function untuk mengukur waktu eksekusi
        start_ts = time.perf_counter()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            raise e
        finally:
            end_ts = time.perf_counter()
            duration = end_ts - start_ts
            if duration > 1.0:
                logger.debug(f"Long execution detected in {func.__name__}: {duration:.4f}s")
    return wrapper


# =============
# GEO MATH CORE
# =============
class GeoMathCore: # Kelas utilitas untuk perhitungan geodesik
    R_EARTH_KM = 6371.0088 # radius bumi dalam kilometer

    # ----------------------------
    # ACO → GA: Lightweight Angle Search
    # ----------------------------
    @staticmethod
    def angle_search_from_aco(center_lat: float, center_lon: float, impact_area_km2: float, #input GA
                              aco_epicenters_csv: Optional[str] = None,
                              sector_half_width_deg: float = 15.0) -> Dict[str, Any]:
        """
        Simple grid-search over 0..359 degrees:
        - score(angle) = sum(pheromone_score) of ACO epicenters that lie inside
          a sector centered at angle (± sector_half_width_deg) and within radius derived from impact_area.
        Returns pred dict (pred_lat, pred_lon, bearing_degree, distance_km, confidence).
        """
        # compute radius from area
        try:
            radius_km = math.sqrt(max(impact_area_km2, 0.0) / math.pi) #Rumus 3.16 & 3.17
            # if area small, use a small default radius
            radius_km = max(radius_km, 1.0)
        except Exception:
            radius_km = 10.0

        # load ACO epicenters (if available)
        if aco_epicenters_csv and os.path.exists(aco_epicenters_csv):
            try:
                df_aco = pd.read_csv(aco_epicenters_csv)
            except Exception:
                df_aco = pd.DataFrame()
        else:
            df_aco = pd.DataFrame()

        # Normalize column names: accept multiple variants
        col_map = {}
        if 'Lintang' in df_aco.columns and 'Bujur' in df_aco.columns:
            col_map['Lintang'] = 'Lintang'
            col_map['Bujur'] = 'Bujur'
        elif 'EQ_Lintang' in df_aco.columns and 'EQ_Bujur' in df_aco.columns:
            col_map['EQ_Lintang'] = 'Lintang'
            col_map['EQ_Bujur'] = 'Bujur'
            df_aco = df_aco.rename(columns={'EQ_Lintang':'Lintang','EQ_Bujur':'Bujur'})
        elif 'Latitude' in df_aco.columns and 'Longitude' in df_aco.columns:
            df_aco = df_aco.rename(columns={'Latitude':'Lintang','Longitude':'Bujur'})

        if df_aco.empty or not {'Lintang','Bujur'}.issubset(set(df_aco.columns)):
            pred_distance = radius_km * 0.5
            pred_lat, pred_lon = GeoMathCore.destination_point(center_lat, center_lon, 0.0, pred_distance)
            return {
                "pred_lat": float(pred_lat),
                "pred_lon": float(pred_lon),
                "bearing_degree": 0.0,
                "distance_km": float(pred_distance),
                "confidence": 0.2
            }


        # ensure pheromone column availability
        pher_col = None
        for c in ['PheromoneScore','Pheromone_Score','Pheromone', 'Risk_Index']:
            if c in df_aco.columns:
                pher_col = c
                break
        if pher_col is None:
            df_aco['__pher__'] = 1.0
            pher_col = '__pher__'
        df_aco[pher_col] = pd.to_numeric(df_aco[pher_col].fillna(0.0), errors='coerce')

        lats = df_aco['Lintang'].astype(float).values
        lons = df_aco['Bujur'].astype(float).values
        phers = df_aco[pher_col].astype(float).values

        best_score = -1.0
        best_angle = 0.0

        # precompute bearings & distances from center to all points
        bearings = np.array([GeoMathCore.calculate_bearing(center_lat, center_lon, la, lo) for la, lo in zip(lats, lons)])
        distances = np.array([GeoMathCore.haversine(center_lat, center_lon, la, lo) for la, lo in zip(lats, lons)])

        for ang in range(0, 360):
            # compute angular diff (0..180)
            dif = np.abs(bearings - ang)
            dif = np.minimum(dif, 360.0 - dif)
            within_sector = dif <= sector_half_width_deg
            within_dist = distances <= radius_km
            mask = within_sector & within_dist
            score = float(np.nansum( phers[mask] ))
            if score > best_score:
                best_score = score
                best_angle = float(ang)

        # --- Compute pred_distance adaptively based on points inside best sector ---
        # mask for best angle sector (repeat the mask calc for best_angle)
        dif = np.abs(bearings - best_angle)
        dif = np.minimum(dif, 360.0 - dif)
        within_sector = dif <= sector_half_width_deg
        within_dist = distances <= radius_km
        mask = within_sector & within_dist

        if np.any(mask):
            # Weighted average distance of points inside sector (weight = pheromone)
            masked_dists = distances[mask]
            masked_phers = phers[mask]
            # avoid divide-by-zero
            wsum = float(np.nansum(masked_phers)) + 1e-12
            weighted_mean_dist = float(np.nansum(masked_dists * masked_phers) / wsum)
            # also consider the farthest point to allow larger reach if pheromone strong
            far_dist = float(np.nanmax(masked_dists))
            # combine mean and far with more weight to mean (tunable)
            pred_distance = float(0.75 * weighted_mean_dist + 0.25 * far_dist)
            # ensure not smaller than a minimum fraction of radius
            pred_distance = max(pred_distance, radius_km * 0.2)
        else:
            # fallback: no points inside sector → use fraction of radius (original behavior)
            pred_distance = radius_km * 0.6

        # compute predicted lat/lon using the adaptive distance
        pred_lat, pred_lon = GeoMathCore.destination_point(center_lat, center_lon, best_angle, pred_distance)

        # confidence: normalized best_score vs total pheromone mass
        total_pher = float(np.nansum(phers)) + 1e-9
        conf = float(min(1.0, best_score / total_pher)) if total_pher > 0 else 0.2

        return {
            "pred_lat": float(pred_lat),
            "pred_lon": float(pred_lon),
            "bearing_degree": float(best_angle),
            "distance_km": float(pred_distance),
            "confidence": float(conf)
        }

    @staticmethod # konversi derajat ke radian
    def to_radians(array_like): # mengonversi array derajat ke radian
        return np.radians(array_like) # menggunakan numpy untuk konversi

    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2): # menghitung bearing antara dua titik
        """Bearing (sudut) dari titik 1 → titik 2 dalam derajat 0-360 (geodesic)."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        diff_lon = math.radians(lon2 - lon1)

        x = math.sin(diff_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(diff_lon))

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360.0) % 360.0

    @classmethod
    def haversine(cls, lat1, lon1, lat2, lon2): # menghitung jarak haversine antara dua titik Rumus 3.20
        """Jarak permukaan bumi (km) antar dua koordinat (great-circle)."""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return cls.R_EARTH_KM * c

    @classmethod
    def destination_point(cls, lat1, lon1, bearing_deg, distance_km): # menghitung titik tujuan berdasarkan bearing dan jarak
        """
        Hitung titik tujuan (lat2, lon2) dari lat1,lon1, bearing (deg) dan distance (km).
        Rumus spherial: lat2 = asin(sin(lat1)*cos(d/R) + cos(lat1)*sin(d/R)*cos(brng))
        """
        if distance_km == 0:
            return float(lat1), float(lon1)

        brng = math.radians(bearing_deg)
        d_div_r = float(distance_km) / cls.R_EARTH_KM

        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)

        lat2_r = math.asin(math.sin(lat1_r) * math.cos(d_div_r) +
                           math.cos(lat1_r) * math.sin(d_div_r) * math.cos(brng))

        lon2_r = lon1_r + math.atan2(math.sin(brng) * math.sin(d_div_r) * math.cos(lat1_r),
                                     math.cos(d_div_r) - math.sin(lat1_r) * math.sin(lat2_r))

        lat2 = math.degrees(lat2_r)
        lon2 = math.degrees(lon2_r)
        # Normalize lon to -180..180
        lon2 = (lon2 + 180) % 360 - 180
        return float(lat2), float(lon2)

    # ------------------ Angle / Cardinal helpers ------------------
    @staticmethod
    def _normalize_angle_deg(angle_deg):
        """Normalize to [0,360). Accept scalar or numpy array."""
        a = np.asarray(angle_deg, dtype=float)
        return (a % 360.0 + 360.0) % 360.0

    @staticmethod
    def _angle_diff_deg(a, b):
        """Minimum absolute difference between angles a and b (deg). scalar or arrays."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        diff = (((a - b + 180.0) % 360.0) - 180.0)
        return np.abs(diff)

    @staticmethod
    def snap_to_cardinal(angle_deg):
        """
        Snap angle(s) to nearest cardinal: Timur(0), Utara(90), Barat(180), Selatan(270).
        Returns (name, angle_deg_cardinal, dev_deg) for scalar input; arrays for vector input.
        """
        one = np.isscalar(angle_deg)
        a = GeoMathCore._normalize_angle_deg(angle_deg)
        card_map = [("Timur", 0.0), ("Utara", 90.0), ("Barat", 180.0), ("Selatan", 270.0)]

        a_flat = np.atleast_1d(a).ravel()
        names = []
        c_angles = []
        devs = []
        for ang in a_flat:
            best_name, best_ca, best_dev = None, None, 1e9
            for nm, ca in card_map:
                d = GeoMathCore._angle_diff_deg(ang, ca)
                if d < best_dev:
                    best_dev = float(d)
                    best_name = nm
                    best_ca = float(ca)
            names.append(best_name)
            c_angles.append(best_ca)
            devs.append(best_dev)
        if one:
            return names[0], float(c_angles[0]), float(devs[0])
        return np.array(names), np.array(c_angles, dtype=float), np.array(devs, dtype=float)

    # -------------------------------------------------
    # Bearing → Compass Direction (STATIC)
    # -------------------------------------------------
    @staticmethod
    def bearing_to_compass_static(bearing: float) -> str:
        """
        Konversi bearing derajat (0–360) ke arah mata angin.
        Dipakai oleh GA untuk write-back ke ACO JSON.
        """
        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ix = int((bearing + 22.5) // 45) % 8
        return dirs[ix]



# ======================
# DATA SANITIZER (REVISI)
# ======================
class DataSanitizer: # Kelas untuk membersihkan dan memvalidasi data input
    def __init__(self):
        # hanya kolom esensial yg wajib ada untuk GA: koordinat + pheromone (opsional)
        self.required_columns = ['EQ_Lintang', 'EQ_Bujur'] 
        self.min_rows = 2  # GA spasial minimal 2 titik untuk komputasi arah

    @execution_monitor
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            raise ValueError("Input DataFrame cannot be None")
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df = df.copy()


        # hanya konversi Acquired_Date jika kolom ada (tidak wajib)
        if 'Acquired_Date' in df.columns:
            try:
                df['Acquired_Date'] = pd.to_datetime(df['Acquired_Date'], errors='coerce')
            except Exception:
                df['Acquired_Date'] = pd.NaT

        # Pastikan koordinat ada dan numeric
        for c in ['EQ_Lintang', 'EQ_Bujur']:
            if c not in df.columns:
                raise ValueError(f"Required column missing: {c}")
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Optional numeric columns: hanya ubah jika ada
        optional_numeric = ['PheromoneScore', 'Magnitudo', 'Kedalaman (km)']
        for c in optional_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

        # Filter lat/lon valid
        df = df.dropna(subset=['EQ_Lintang', 'EQ_Bujur']).reset_index(drop=True)
        df = df[(df['EQ_Lintang'] >= -90) & (df['EQ_Lintang'] <= 90)]
        df = df[(df['EQ_Bujur'] >= -180) & (df['EQ_Bujur'] <= 180)]
        df = df.reset_index(drop=True)

        if len(df) < self.min_rows:
            raise ValueError(f"Min rows not met: {len(df)} (need >= {self.min_rows})")

        # Jika PheromoneScore tidak ada, set default kecil supaya risk-based logic aman
        if 'PheromoneScore' not in df.columns:
            df['PheromoneScore'] = 0.1

        return df



# ======================
# PHYSICS FITNESS ENGINE (REVISI: SPATIAL ONLY)
# ======================
class PhysicsFitnessEngine: 
    def __init__(self, df: pd.DataFrame, weight_config: Dict[str, float]):
        self.df = df
        self.weights = weight_config

        # --- REVISI FATAL: Hapus Load Data Waktu & Magnitudo ---
        # Sesuai Diagram Blok: Input hanya Koordinat & Pheromone (Risk)
        self.vec_lat = df['EQ_Lintang'].values
        self.vec_lon = df['EQ_Bujur'].values
        
        # Pheromone/Risk dari ACO dipakai sebagai bobot utama
        self.vec_risk = df['PheromoneScore'].values

        self.num_nodes = len(df)
        self.penalty = 1e15

        # --- REVISI: Hapus Bobot Waktu (w_time) ---
        # Kita hanya fokus pada Spasial (Jarak) dan Risiko (Pheromone)
        self.w_space = weight_config.get("space", 1.0)
        self.w_risk = weight_config.get("risk", 500.0)

    def evaluate(self, individual: List[int]) -> Tuple[float]:
        """
        Fungsi Evaluasi (Fitness Function) - MURNI SPASIAL
        
        KOREKSI PENGUJI:
        Tidak ada variabel 'c_time' atau 'Acquired_Date'.
        Hanya menghitung Jarak (Space) dan Risiko (Risk).
        
        Rumus: F = (w_space * Total_Jarak) + (w_risk * Total_Inverse_Risk)
        """
        idx = np.array(individual, dtype=int)

        if len(idx) != self.num_nodes:
            return (self.penalty, )

        # Ambil data lookup
        curr_risks = self.vec_risk[idx]
        lat_seq = self.vec_lat[idx]
        lon_seq = self.vec_lon[idx]

        # 1. SPATIAL COST (Jarak Fisik)
        # Menghitung total jarak haversine antar titik dalam kromosom
        total_dist = 0.0
        for i in range(len(idx) - 1):
            total_dist += GeoMathCore.haversine(
                lat_seq[i], lon_seq[i], #(t-1)
                lat_seq[i+1], lon_seq[i+1] #(t)
            )
        spatial_cost = self.w_space * total_dist #Rumus 3.19

        # 2. RISK SCORE COST (Pheromone Optimization)
        # GA mencari jalur yang melewati titik dengan Pheromone TINGGI.
        # Karena Fitness = Minimasi Cost, maka Risk harus di-invers (1/Risk).
        # Semakin tinggi Risk, semakin kecil Cost (Bagus).
        clipped_risk = np.clip(curr_risks, 1e-6, None) # Hindari pembagian nol
        risk_cost = self.w_risk * np.sum(1.0 / clipped_risk) #Rumus 3.21

        # --- REVISI: Temporal Cost DIHAPUS ---
        # Code lama menghitung pelanggaran urutan waktu.
        # Bagian itu SUDAH DIHAPUS sesuai request penguji.
        
        # 3. TOTAL COST
        total_cost = spatial_cost + risk_cost #Rumus 3.18
        
        if np.isnan(total_cost) or np.isinf(total_cost):
            return (self.penalty, )

        return (total_cost, )


    # -------------------------------------------------------------
    # PREDIKSI NEXT EVENT (Vektor Arah)
    # -------------------------------------------------------------
    def predict_next_event(self, df_path: pd.DataFrame, n_seg: Optional[int] = 5) -> Dict[str, float]:
            if df_path is None or len(df_path) < 2:
                return {
                    "pred_lat": 0.0, "pred_lon": 0.0,
                    "bearing_degree": 0.0, "distance_km": 0.0, "confidence": 0.0
                }

            if n_seg is None:
                n_seg = min(5, len(df_path))
            else:
                n_seg = min(max(2, int(n_seg)), len(df_path))

            df_seg = df_path.iloc[-n_seg:].reset_index(drop=True)
            return self.compute_vector_from_segment(df_seg)

    def compute_vector_from_segment(self, df_seg: pd.DataFrame) -> Dict[str, float]:
        """
        Menghitung vektor rata-rata dari jalur terbaik (Best Individual)
        """
        if df_seg is None or len(df_seg) < 2:
            return {}

        lats = df_seg['EQ_Lintang'].astype(float).values
        lons = df_seg['EQ_Bujur'].astype(float).values
        risks = df_seg['PheromoneScore'].astype(float).values if 'PheromoneScore' in df_seg.columns else np.ones(len(df_seg)) * 0.1

        bearings = []
        distances = []
        weights = []

        for i in range(len(lats) - 1):
            lat_a, lon_a = lats[i], lons[i] #t-1
            lat_b, lon_b = lats[i + 1], lons[i + 1] #t
            
            dkm = GeoMathCore.haversine(lat_a, lon_a, lat_b, lon_b)
            bdeg = GeoMathCore.calculate_bearing(lat_a, lon_a, lat_b, lon_b)

            # Weight vector berdasarkan Pheromone rata-rata antar dua titik
            w = ((risks[i] + risks[i + 1]) / 2.0) + 1e-9

            distances.append(float(dkm))
            bearings.append(float(bdeg))
            weights.append(float(w))

        distances = np.array(distances, dtype=float)
        weights = np.array(weights, dtype=float)
        weights = weights / (np.sum(weights) + 1e-12)

        # Circular mean for bearing
        thetas = np.radians(np.array(bearings))
        x = float(np.sum(weights * np.cos(thetas)))
        y = float(np.sum(weights * np.sin(thetas)))
        mean_theta = math.atan2(y, x) if not (x == 0 and y == 0) else 0.0 #Rumus 3.23
        mean_bearing_deg = (math.degrees(mean_theta) + 360.0) % 360.0

        # Mean Distance
        mean_distance_km = float(np.sum(distances * weights)) if distances.size > 0 else 0.0
        
        # Scale (Revisi: hanya gunakan risk, tanpa magnitudo/waktu)
        last_risk = float(risks[-1]) if len(risks) > 0 else 0.1
        scale = 0.5 + float(last_risk)
        pred_distance_km = float(mean_distance_km * scale)

        pred_lat, pred_lon = GeoMathCore.destination_point(float(lats[-1]), float(lons[-1]), mean_bearing_deg, pred_distance_km)

        # Confidence calculation
        conf_risk = self.compute_confidence(df_seg)
        
        # --- KONVERSI KE ARAH MATA ANGIN (CARDINAL) ---
        # Ini menjawab poin client: Sudut -> Arah Mata Angin
        def _angle_diff_deg(a, b):
            return abs(((a - b + 180.0) % 360.0) - 180.0)

        card_map = [("Timur", 0.0), ("Utara", 90.0), ("Barat", 180.0), ("Selatan", 270.0)]
        best_name, best_card_deg, best_dev = None, None, 1e9
        for nm, ca in card_map:
            d = _angle_diff_deg(mean_bearing_deg, ca)
            if d < best_dev:
                best_dev = float(d)
                best_name = nm
                best_card_deg = float(ca)

        if best_name is None:
            best_name = "N/A"

        return {
            "pred_lat": float(pred_lat),
            "pred_lon": float(pred_lon),
            "base_lat": float(lats[-1]),
            "base_lon": float(lons[-1]),
            "movement_scale": float(scale),
            "bearing_degree": float(mean_bearing_deg), # Solusi Terbaik Sudut
            "distance_km": float(pred_distance_km),
            "movement_direction": best_name, # Konversi Arah Mata Angin
            "confidence": float(conf_risk)
        }

    @staticmethod
    def bearing_to_compass(bearing: float) -> str:
        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ix = int((bearing + 22.5) // 45) % 8
        return dirs[ix]

    def compute_confidence(self, df_seg: pd.DataFrame) -> float:
        try:
            risks = df_seg['PheromoneScore'].astype(float).values
            if np.all(risks == 0): return 0.4
            mean_r = float(np.mean(risks))
            std_r = float(np.std(risks)) + 1e-6
            conf = mean_r / (mean_r + std_r)
            return max(0.3, min(conf, 1.0))
        except Exception: 
            return 0.4

# ============================================
# BLOCK 2/3 — Evolutionary Controller + Checkpoint
# ============================================
# Checkpoint System
class CheckpointSystem:
    def __init__(self, directory: str, filename: str = "ga_state.pkl"): # inisialisasi sistem checkpoint
        self.directory = os.path.join(directory, "checkpoints") # direktori penyimpanan checkpoint
        self.filename = filename # nama file checkpoint default
        self.filepath = os.path.join(self.directory, filename) # path lengkap file checkpoint 
        os.makedirs(self.directory, exist_ok=True) # buat direktori jika belum ada
    # simpan state ke file checkpoint
    def save_state(self, population, generation, stats, filename=None): # simpan state ke file checkpoint
        fname = filename if filename else self.filename # gunakan nama file khusus jika diberikan
        fpath = os.path.join(self.directory, fname) # path lengkap file checkpoint

        payload = {
            "version": "6.0",
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "population": population,
            "stats": stats
        } # data yang akan disimpan
        # simpan ke file menggunakan pickle
        try:
            with open(fpath, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    # muat state dari file checkpoint
    def load_state(self):
        if not os.path.exists(self.filepath):
            return None
        try:
            with open(self.filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

# Statistics Collector untuk GA Engine 
class StatisticsCollector:
    def __init__(self):
        self.stats = tools.Statistics(lambda ind: ind.fitness.values) # mengumpulkan nilai fitness individu
        self.stats.register("avg", np.mean) # rata-rata fitness
        self.stats.register("std", np.std)# standar deviasi fitness
        self.stats.register("min", np.min) # nilai fitness minimum
        self.stats.register("max", np.max)  # nilai fitness maksimum
    # dapatkan objek statistik
    def get(self):
        return self.stats

# Evolutionary Controller untuk menjalankan GA
class EvolutionaryController:
    def __init__(self, config, fitness_engine, checkpoint_mgr): # inisialisasi controller evolusi
        self.cfg = config # konfigurasi GA
        self.fitness_engine = fitness_engine # engine fitness
        self.checkpoint_mgr = checkpoint_mgr # manajer checkpoint

        self.toolbox = base.Toolbox() # toolbox DEAP untuk operator GA
        self.stats_col = StatisticsCollector() # kolektor statistik
        self.stats = self.stats_col.get() # objek statistik
        # daftarkan operator GA
        self._register_operators()
    # daftarkan operator GA
    def _register_operators(self):
        problem_size = self.fitness_engine.num_nodes # ukuran masalah (jumlah node)

        # Index Generator (Perumutan) untuk kromosom permutasi 
        self.toolbox.register("indices", np.random.permutation, problem_size)

        # Chromosome Initialization
        def init_chromosome(icls, generator):
            return icls(generator().tolist())

        self.toolbox.register("individual", init_chromosome,
                              creator.Individual, self.toolbox.indices)

        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        # Fitness
        self.toolbox.register("evaluate", self.fitness_engine.evaluate)

        # Operators
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes,
                              indpb=self.cfg.mutation_prob)
        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.cfg.tournament_size)

    @execution_monitor # dekorator untuk memonitor eksekusi
    def run(self): # metode utama untuk menjalankan evolusi GA
        pop_size = self.cfg.population_size # ukuran populasi
        n_gen = self.cfg.n_generations # jumlah generasi
        cx_prob = self.cfg.crossover_prob # probabilitas crossover
        mut_prob = self.cfg.mutation_prob # probabilitas mutasi
        # Logging awal
        logger.info(f"GA Evolution Start → Pop={pop_size}, Gen={n_gen}")

        
        pop = self.toolbox.population(n=pop_size) # inisialisasi populasi
        hof = tools.HallOfFame(self.cfg.hall_of_fame_size) # hall of fame untuk individu terbaik

        # Load checkpoint if exists
        state = self.checkpoint_mgr.load_state()
        if state:
            logger.info("Resuming GA from checkpoint...")
            pop = state["population"]
            start_gen = state["generation"]
        else:
            start_gen = 0

        # Initial evaluation
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = map(self.toolbox.evaluate, invalid)

        for ind, fval in zip(invalid, fits):
            ind.fitness.values = fval

        hof.update(pop)

        # Statistics logbook
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + self.stats.fields

        record = self.stats.compile(pop)
        logbook.record(gen=start_gen, nevals=len(invalid), **record)

        best_so_far = record["min"]
        stagnation = 0

        # GENERATIONAL LOOP
        for gen in range(start_gen + 1, n_gen + 1):

            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Adaptive Mutation
            adaptive_mut = mut_prob
            if stagnation > 20:
                adaptive_mut = min(0.9, mut_prob * 2.0)

            for mutant in offspring:
                if random.random() < adaptive_mut:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate new individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            newfits = map(self.toolbox.evaluate, invalid)
            for ind, fval in zip(invalid, newfits):
                ind.fitness.values = fval

            # Replace old population
            pop[:] = offspring
            hof.update(pop)

            # Logging statistics
            rec = self.stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid), **rec)

            # Stagnation check
            if rec["min"] < best_so_far:
                best_so_far = rec["min"]
                stagnation = 0
            else:
                stagnation += 1

            # Save every 10 generations
            if gen % 10 == 0:
                fname = f"ga_state_gen_{gen}.pkl"
                self.checkpoint_mgr.save_state(
                    pop,
                    gen,
                    logbook,
                    filename=fname
                )

        # Save final state
        self.checkpoint_mgr.save_state(pop, n_gen, logbook, filename="final_ga.pkl")

        best_individual = list(hof[0]) #Rumus 3.22
        log_df = pd.DataFrame(logbook)

        return best_individual, log_df, hof


# ============================================
# BLOCK 3/3 — Visualizer + Exporter + GA Engine Wrapper
# ============================================
# Multi-Layer Visualizer untuk peta interaktif
class MultiLayerVisualizer:
    def __init__(self, output_dir: str): 
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visuals")
        os.makedirs(self.viz_dir, exist_ok=True)

    def clamp_to_east_java(self, lat: float, lon: float) -> Tuple[float, float]: 
        LAT_MIN, LAT_MAX = -9.5, -5.5 
        LON_MIN, LON_MAX = 110.0, 116.0 
        clamped_lat = max(min(lat, LAT_MAX), LAT_MIN)
        clamped_lon = max(min(lon, LON_MAX), LON_MIN)
        return clamped_lat, clamped_lon 

    def generate_map(
        self,
        best_path: List[int],
        df: pd.DataFrame,
        pred_info: Dict[str, Any],
        out_path: str,
        ga_vectors: Optional[List[Dict[str, Any]]] = None
    ):

        # =========================
        # MODE DETECTION
        # =========================
        is_aco_mode = bool(ga_vectors)

        # --- LOGIC FIX: Penentuan Titik Pusat Peta (Center) ---
        # Prioritas 1: Gunakan ACO Center (jika Mode ACO)
        center_lat = pred_info.get("aco_center_lat")
        center_lon = pred_info.get("aco_center_lon")

        # Prioritas 2: Gunakan Base Location (Titik terakhir data, jika Mode GA Standard)
        if center_lat is None or center_lon is None:
            center_lat = pred_info.get("base_lat")
            center_lon = pred_info.get("base_lon")

        # Prioritas 3: Gunakan Prediksi Lokasi
        if center_lat is None or center_lon is None:
            center_lat = pred_info.get("pred_lat")
            center_lon = pred_info.get("pred_lon")

        # Prioritas 4: Gunakan Rata-rata Data (Fallback terakhir)
        if (center_lat is None or center_lon is None) and not df.empty:
            center_lat = df['EQ_Lintang'].mean()
            center_lon = df['EQ_Bujur'].mean()

        # Jika masih gagal (Data kosong dan tidak ada prediksi), hentikan pembuatan peta tanpa error fatal
        if center_lat is None or center_lon is None:
            logger.warning("Map center could not be determined. Skipping map generation.")
            return

        if best_path is None:
            best_path = []

        # Buat objek peta folium
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles=None)
        folium.TileLayer('CartoDB positron', name='Light').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)

        # CHAOS LAYER
        chaos = folium.FeatureGroup(name="Chaos Connectivity", show=False)
        coords = df[['EQ_Lintang', 'EQ_Bujur']].values
        if len(coords) > 0:
            sample = coords[: min(400, len(coords))]
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    dist = GeoMathCore.haversine(
                        sample[i][0], sample[i][1],
                        sample[j][0], sample[j][1]
                    )
                    if dist < 40:
                        folium.PolyLine(
                            [sample[i], sample[j]],
                            color="cyan", weight=0.5, opacity=0.25
                        ).add_to(chaos)

            for r in df.itertuples():
                folium.CircleMarker(
                     [getattr(r, 'EQ_Lintang'), getattr(r, 'EQ_Bujur')],
                        radius=3,
                        color="yellow",
                        fill=True,
                        fill_color="yellow",
                        fill_opacity=0.6
                ).add_to(chaos)

        chaos.add_to(m)

        # SNAKE LAYER (Best GA path)
        snake = folium.FeatureGroup(name="Snake Path (Best Sequence)", show=True)

        max_idx = len(df) - 1
        safe_path = [i for i in best_path if 0 <= i <= max_idx]

        # Jika TIDAK mode ACO, baru boleh fallback snake path
        if not safe_path and not df.empty and not is_aco_mode:
            safe_path = list(range(max(0, len(df)-5), len(df)))

        # GA STANDARD ONLY (1 VECTOR)
        if pred_info and "bearing_degree" in pred_info and not ga_vectors:
            best_df = df.iloc[safe_path].reset_index(drop=True)
            path_coords = best_df[['EQ_Lintang', 'EQ_Bujur']].values.tolist()

            if len(path_coords) >= 2:
                AntPath(
                    locations=path_coords,
                    color="magenta", pulse_color="yellow", weight=4, delay=600
                ).add_to(snake)

            if len(path_coords) >= 1:
                folium.Marker(path_coords[0], icon=folium.Icon(color="green", icon="play"), popup="START").add_to(snake)
                folium.Marker(path_coords[-1], icon=folium.Icon(color="red", icon="stop"), popup="END").add_to(snake)

            # Popup info
            for i in range(len(best_df)):
                row = best_df.iloc[i]
                lat = float(row.get('EQ_Lintang', float('nan')))
                lon = float(row.get('EQ_Bujur', float('nan')))
                
                # ... (Perhitungan popup tetap sama) ...
                depth_val = row.get('Kedalaman (km)', row.get('depth', None))
                depth_str = f"{depth_val}" if depth_val is not None else "N/A"
                
                popup = f"""
                <b>Event Detail</b><br>
                Date: {row.get('Acquired_Date', 'N/A')}<br>
                Mag: {row.get('Magnitudo', 'N/A')}<br>
                Depth: {depth_str} km<br>
                Risk: {row.get('PheromoneScore', 0):.3f}<br>
                """
                folium.CircleMarker(
                    [lat, lon], radius=4, color="orange", fill=True, fill_color="yellow", fill_opacity=0.8,
                    popup=folium.Popup(popup, max_width=320)
                ).add_to(snake)

        snake.add_to(m)

        # DIRECTION VISUALIZATION
        if pred_info and "bearing_degree" in pred_info:
            start_lat = center_lat
            start_lon = center_lon

            bearing = pred_info.get("bearing_degree")
            distance_km = pred_info.get("distance_km", 3.0)

            end_lat, end_lon = GeoMathCore.destination_point(
                start_lat, start_lon, bearing, distance_km
            )

            popup_dir = f"<b>Predicted Direction</b><br>Bearing: {bearing:.1f}°"

            # Draw main ray
            folium.PolyLine(
                [[start_lat, start_lon], [end_lat, end_lon]],
                color="orange", weight=4, opacity=0.8, tooltip="Prediction Vector"
            ).add_to(m)

            folium.Marker(
                [end_lat, end_lon],
                popup=folium.Popup(popup_dir, max_width=280),
                icon=folium.DivIcon(html="<div style='font-size:20px'>➤</div>")
            ).add_to(m)

            # Draw radius circle and spokes (like ACO case)
            folium.Circle(
                location=[start_lat, start_lon],
                radius=max(100.0, distance_km * 1000.0),
                color="orange",
                weight=2,
                opacity=0.7,
                fill=True,
                fill_opacity=0.06,
                popup=folium.Popup(f"Predicted radius {distance_km:.2f} km", max_width=240)
            ).add_to(m)

            # spokes
            for s in range(12):  # finer spokes for single vector
                b_spoke = s * (360.0 / 12)
                sp_lat, sp_lon = GeoMathCore.destination_point(start_lat, start_lon, b_spoke, distance_km)
                folium.PolyLine(
                    [[start_lat, start_lon], [sp_lat, sp_lon]],
                    color="green",
                    weight=1,
                    opacity=0.4
                ).add_to(m)


        # =========================
        # GA VECTOR FROM ACO
        # =========================
        if ga_vectors:
            ga_layer = folium.FeatureGroup(name="GA Direction (Per ACO Node)", show=True)

            # track centers already drawn to avoid duplicate circles
            seen_centers = set()

            for v in ga_vectors:
                start_lat = float(v["base_lat"])
                start_lon = float(v["base_lon"])
                bearing = float(v["bearing_degree"])
                dist = float(v["distance_km"])  # in km

                # End point (existing behavior)
                end_lat, end_lon = GeoMathCore.destination_point(
                    start_lat, start_lon, bearing, dist
                )

                # 1) Draw the main vector ray (kept as-is)
                folium.PolyLine(
                    [[start_lat, start_lon], [end_lat, end_lon]],
                    color="red",
                    weight=4,
                    opacity=0.95
                ).add_to(ga_layer)

                # 2) Draw arrow marker at end (kept as-is)
                folium.Marker(
                    [end_lat, end_lon],
                    icon=folium.DivIcon(
                        html="<div style='font-size:20px;color:red'>➤</div>"
                    ),
                    popup=f"""
                    <b>GA Vector</b><br>
                    From ACO index: {v['from_aco_index']}<br>
                    Bearing: {bearing:.1f}°<br>
                    Distance: {dist:.2f} km
                    """
                ).add_to(ga_layer)

                # 3) Draw circle (radius = dist km) centered at start (non-filled, semi-transparent)
                # Use tuple rounded as key to prevent duplicate circles for same center+radius
                center_key = (round(start_lat, 6), round(start_lon, 6), round(dist, 3))
                if center_key not in seen_centers:
                    seen_centers.add(center_key)

                    folium.Circle(
                        location=[start_lat, start_lon],
                        radius=max(100.0, dist * 1000.0),
                        color="green",
                        weight=2,
                        opacity=0.8,
                        fill=True,
                        fill_color="green",
                        fill_opacity=0.12,
                        popup=folium.Popup(f"Radius {dist:.2f} km from ACO idx {v['from_aco_index']}", max_width=240)
                    ).add_to(ga_layer)

                    # 4) Optional: Draw radial spokes (jari-jari)
                    #    - number of spokes bisa disesuaikan (default 8)
                    # 4) Radial spokes (jari-jari area GA)
                    n_spokes = 8
                    for s in range(n_spokes):
                        b_spoke = s * (360.0 / n_spokes)
                        sp_lat, sp_lon = GeoMathCore.destination_point(start_lat, start_lon, b_spoke, dist)
                        folium.PolyLine(
                            [[start_lat, start_lon], [sp_lat, sp_lon]],
                            color="green",
                            weight=1,
                            opacity=0.4
                        ).add_to(ga_layer)

            ga_layer.add_to(m)



        # Heatmap
        try:
            if 'PheromoneScore' in df.columns:
                heat_data = df[['EQ_Lintang', 'EQ_Bujur', 'PheromoneScore']].values.tolist()
                HeatMap(heat_data, name="Risk Heatmap", radius=15, blur=10, show=False).add_to(m)
        except Exception: pass

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MiniMap().add_to(m)

        from tempfile import NamedTemporaryFile
        dirn = os.path.dirname(out_path) or "."
        with NamedTemporaryFile("w", dir=dirn, delete=False, suffix=".html", encoding="utf-8") as tf:
            tmpname = tf.name
            m.save(tmpname)
        os.replace(tmpname, out_path)
        logger.info(f"GA Map saved (atomic) → {out_path}")

# Excel Data Exporter
class DataExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.excel_path = os.path.join(output_dir, "ga_report.xlsx")

    def export(self,
               df_original: pd.DataFrame,
               df_optimal: pd.DataFrame,
               meta: Dict[str, Any],
               ga_input: Optional[Dict[str, Any]] = None,
               ga_output: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None):
        """
        Export sheets:
          - RawData
          - BestPath
          - Meta
          - GA_Input (center + impact area)
          - GA_Output (angle + direction)
        """
        try:
            os.makedirs(os.path.dirname(self.excel_path), exist_ok=True)
            with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
                # Raw / cleaned data
                try:
                    df_original.to_excel(writer, sheet_name="RawData", index=False)
                except Exception:
                    pd.DataFrame(df_original).to_excel(writer, sheet_name="RawData", index=False)

                # Best path / selection
                try:
                    df_optimal.to_excel(writer, sheet_name="BestPath", index=False)
                except Exception:
                    pd.DataFrame(df_optimal).to_excel(writer, sheet_name="BestPath", index=False)

                # Meta
                pd.DataFrame([meta]).to_excel(writer, sheet_name="Meta", index=False)

                # GA input: center & area
                if ga_input:
                    pd.DataFrame([ga_input]).to_excel(writer, sheet_name="GA_Input", index=False)

                # GA output: list of dicts or single dict
                if ga_output:
                    if isinstance(ga_output, dict):
                        pd.DataFrame([ga_output]).to_excel(writer, sheet_name="GA_Output", index=False)
                    else:
                        pd.DataFrame(ga_output).to_excel(writer, sheet_name="GA_Output", index=False)

            logger.info(f"Excel exported → {self.excel_path}")
        except Exception as e:
            logger.error(f"Excel export failed: {e}", exc_info=True)

# ==================================================
# GA ENGINE WRAPPER (Dipanggil Pipeline)
# ==================================================
class GaEngine: # GA Engine utama
    def __init__(self, config: Any): # inisialisasi GA Engine
        self.cfg = config # konfigurasi GA
        self.output_dir = getattr(config, "output_dir", "output/ga_results") # direktori output
        os.makedirs(self.output_dir, exist_ok=True) # buat direktori output jika belum ada

        # MAPS directory (UNTUK FOLIUM MAP)
        self.maps_dir = os.path.join(self.output_dir, "maps")
        os.makedirs(self.maps_dir, exist_ok=True)

        # Path FINAL map
        self.map_path = os.path.join(self.maps_dir, "ga_path_map.html")

        self.sanitizer = DataSanitizer() # inisialisasi data sanitizer
        self.checkpoint_mgr = CheckpointSystem(self.output_dir) # inisialisasi manajer checkpoint
        self.visualizer = MultiLayerVisualizer(self.output_dir) # inisialisasi visualizer peta
        self.exporter = DataExporter(self.output_dir)

        self.map_path = os.path.join(self.output_dir, "ga_path_map.html")
        self.log_path = os.path.join(self.output_dir, "ga_log.csv")

    def _attach_ga_targets(
        self,
        df: pd.DataFrame,
        pred: Dict[str, Any]
    ) -> pd.DataFrame:

        if df is None or df.empty:
            raise ValueError("df_train kosong saat attach GA targets")

        required = ["bearing_degree", "distance_km"]
        for k in required:
            if k not in pred or pred[k] is None:
                raise RuntimeError(f"GA output missing required key: {k}")

        df = df.copy()
        df["ga_bearing_deg"] = float(pred["bearing_degree"]) #output GA
        df["ga_distance_km"] = float(pred["distance_km"])
        df["ga_confidence"] = float(pred.get("confidence", 0.0))

        return df

    # Tulis hasil prediksi ke aco_to_ga.json
    def _write_back_to_aco_json(self, pred: Dict[str, Any]):
        """
        GA OUTPUT (client-compliant):
        - HANYA arah (teks) dan sudut pergerakan (derajat)
        - Tidak ada lokasi, magnitudo, confidence, atau distance.
        """
        import os
        import json
        import numpy as np
        from datetime import datetime
        from tempfile import NamedTemporaryFile

        aco_json_path = os.path.join(
            os.path.dirname(self.output_dir),
            "aco_results",
            "aco_to_ga.json"
        )

        os.makedirs(os.path.dirname(aco_json_path), exist_ok=True)

        def safe_float(x, default=None):
            try:
                fx = float(x)
                return fx if np.isfinite(fx) else default
            except Exception:
                return default

        # Prepare minimal payload expected by client: sudut + arah teks
        bearing = safe_float(pred.get("bearing_degree"), default=None)
        # fallback compute textual direction if not present
        move_dir = pred.get("movement_direction")
        if move_dir is None and bearing is not None:
            move_dir = GeoMathCore.bearing_to_compass_static(bearing) if hasattr(GeoMathCore, "bearing_to_compass_static") else self.visualizer_bearing_to_compass(bearing)

        next_event = {}
        if bearing is not None:
            next_event["angle_deg"] = bearing
        if move_dir is not None:
            next_event["direction"] = move_dir

        # If neither present, log & do not overwrite ACO output
        if not next_event:
            logger.warning("[GA] No valid bearing/direction to write. Skipping write-back.")
            return

        try:
            # load existing ACO json (preserve its other fields)
            if os.path.exists(aco_json_path):
                with open(aco_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            # update only next_event key
            data["next_event"] = next_event
            data["_ga_generated_at"] = datetime.now().isoformat()

            # atomic write (temp file + rename)
            dirn = os.path.dirname(aco_json_path)
            with NamedTemporaryFile("w", dir=dirn, delete=False, encoding="utf-8") as tf:
                json.dump(data, tf, indent=2, ensure_ascii=False)
                temp_name = tf.name
            os.replace(temp_name, aco_json_path)

            logger.info("[GA] next_event (direction-only) ditulis ke aco_to_ga.json")
        except Exception:
            logger.error("[GA] Gagal menulis next_event GA", exc_info=True)


    # Metode utama untuk menjalankan GA Engine
    def run(self, df_train: pd.DataFrame) -> Tuple[List[int], Dict[str, Any]]:
        logger.info("\n" + "=" * 80)
        logger.info("=== GA ENGINE START (Input from ACO Only) ===".center(80))
        logger.info("=" * 80)

        # --- LOGIKA REVISI: INPUT DARI OUTPUT ACO AJA ---
        aco_results_dir = os.path.join(os.path.dirname(self.output_dir), "aco_results")
        aco_json_path = os.path.join(aco_results_dir, "aco_to_ga.json")
        aco_epicenters_csv = os.path.join(aco_results_dir, "aco_epicenters.csv")

        # Cek apakah ACO output tersedia
        if os.path.exists(aco_json_path):
            try:
                with open(aco_json_path, "r", encoding="utf-8") as f:
                    aco_payload = json.load(f)
                
                # Input GA: Pusat ACO dan Area ACO
                impact_area = float(
                    aco_payload.get("impact_area_km2", aco_payload.get("impact_area", 0.0)) #Rumus 3.14
                )

                ga_vectors = []

                # Baca titik-titik ACO
                df_aco = pd.read_csv(aco_epicenters_csv)

                # Normalisasi nama kolom
                rename_map = {
                    'Lintang': 'EQ_Lintang', 'Bujur': 'EQ_Bujur',
                    'Latitude': 'EQ_Lintang', 'Longitude': 'EQ_Bujur'
                }
                df_aco.rename(columns=rename_map, inplace=True)

                for idx, row in df_aco.iterrows():
                    base_lat = float(row["EQ_Lintang"])
                    base_lon = float(row["EQ_Bujur"])

                    # Jalankan GA dari TITIK ACO ini
                    pred = GeoMathCore.angle_search_from_aco(
                        base_lat,
                        base_lon,
                        impact_area,
                        aco_epicenters_csv
                    )

                    ga_vectors.append({
                        "from_aco_index": int(idx),
                        "base_lat": base_lat,
                        "base_lon": base_lon,
                        "bearing_degree": float(pred.get("bearing_degree", 0.0)),
                        "distance_km": float(pred.get("distance_km", 0.0))
                    })
                # =========================================================
                # GUARD PENTING: pastikan pred selalu terdefinisi
                # =========================================================
                if not ga_vectors:
                    logger.warning("[GA] No ACO epicenters found; using default GA prediction.")
                    pred = {
                        "bearing_degree": 0.0,
                        "distance_km": 0.0
                    }
                else:
                    # Ambil vektor representatif (misalnya dari ACO pertama)
                    pred = {
                        "bearing_degree": ga_vectors[0]["bearing_degree"],
                        "distance_km": ga_vectors[0]["distance_km"]
                    }
                # =====================================================
                # [OPSI 1] TIME-AWARE GA EVENT (ALWAYS EXECUTED)
                # =====================================================

                ga_event_time = None
                if "timestamp" in aco_payload:
                    ga_event_time = pd.to_datetime(aco_payload["timestamp"], errors="coerce")

                if ga_event_time is None or pd.isna(ga_event_time):
                    ga_event_time = pd.Timestamp.utcnow()

                ga_event = {
                    "timestamp": ga_event_time,
                    "ga_bearing_deg": float(pred["bearing_degree"]),
                    "ga_distance_km": float(pred["distance_km"]),
                    "ga_direction": GeoMathCore.bearing_to_compass_static(pred["bearing_degree"])
                }

                try:
                    ga_event_path = os.path.join(self.output_dir, "ga_events.csv")

                    df_evt = pd.DataFrame([ga_event])

                    if os.path.exists(ga_event_path):
                        df_old = pd.read_csv(ga_event_path, parse_dates=["timestamp"])
                        df_all = pd.concat([df_old, df_evt], ignore_index=True)
                    else:
                        df_all = df_evt

                    df_all.sort_values("timestamp", inplace=True)
                    df_all.to_csv(ga_event_path, index=False)

                    logger.info(f"[GA] Event GA tersimpan → {ga_event_path}")

                except Exception as e:
                    logger.warning(f"[GA] Gagal menyimpan GA event: {e}")

                # =====================================================
                # WRITE BACK GA PATH (PER TITIK ACO)
                # =====================================================
                ga_path = []
                for v in ga_vectors:
                    ga_path.append({
                        "from_aco": v["from_aco_index"],
                        "angle_deg": v["bearing_degree"],
                        "direction": GeoMathCore.bearing_to_compass_static(v["bearing_degree"]),
                        "distance_km": v["distance_km"]
                    })

                aco_payload["ga_path"] = ga_path
                aco_payload["_ga_generated_at"] = datetime.now().isoformat()

                with open(aco_json_path, "w", encoding="utf-8") as f:
                    json.dump(aco_payload, f, indent=2)

                # Simpan vektor untuk LSTM
                try:
                    if ga_vectors:
                        vector_out = {
                            "_generated_at": datetime.now().isoformat(),
                            "bearing_degree": ga_vectors[0]["bearing_degree"],
                            "distance_km": ga_vectors[0]["distance_km"],
                            "note": "Representative GA vector (first ACO node)"
                        }

                    with open(os.path.join(self.output_dir, "ga_vector.json"), "w", encoding="utf-8") as vf:
                        json.dump(vector_out, vf, indent=2)
                except Exception as e:
                    logger.warning(f"[GA] Failed to save ga_vector.json: {e}")

                # ---------------------------
                # Export GA Excel (ACO mode)
                # ---------------------------
                try:
                    # GA input (center + impact area) from aco_payload
                    ga_input = {
                        "center_lat": float(aco_payload.get("center_lat", aco_payload.get("center_lat", None) or 0.0)),
                        "center_lon": float(aco_payload.get("center_lon", aco_payload.get("center_lon", None) or 0.0)),
                        "impact_area_km2": float(impact_area)
                    }

                    # GA output (list of per-ACO-node directions)
                    ga_output = ga_path  # ga_path dibuat sebelumnya: list of dicts with angle_deg & direction

                    # Export: use df_aco as RawData; BestPath left empty for pure ACO->GA mode
                    self.exporter.export(
                        df_original = df_aco.reset_index(drop=True),
                        df_optimal = pd.DataFrame(),  # no GA permutation in this mode
                        meta = {
                            "mode": "ACO->GA",
                            "timestamp": datetime.now().isoformat(),
                            "note": "GA vectors per ACO epicenter"
                        },
                        ga_input = ga_input,
                        ga_output = ga_output
                    )
                except Exception as e:
                    logger.warning(f"[GA] Excel export (ACO mode) failed: {e}", exc_info=True)

                # --- FIX 1: Generate Map dengan Handling Nama Kolom ---
                try:
                    if os.path.exists(aco_epicenters_csv):
                        df_aco = pd.read_csv(aco_epicenters_csv)
                        
                        # PERBAIKAN: Rename kolom Lintang/Bujur ke EQ_Lintang/EQ_Bujur
                        # Ini mengatasi error KeyError saat Visualizer membaca file CSV ACO
                        rename_map = {
                            'Lintang': 'EQ_Lintang', 'Bujur': 'EQ_Bujur',
                            'Latitude': 'EQ_Lintang', 'Longitude': 'EQ_Bujur'
                        }
                        df_aco.rename(columns=rename_map, inplace=True)
                        
                        # Fallback ekstra jika rename tidak lengkap
                        if 'EQ_Lintang' not in df_aco.columns and 'Lintang' in df_aco.columns:
                            df_aco['EQ_Lintang'] = df_aco['Lintang']
                        if 'EQ_Bujur' not in df_aco.columns and 'Bujur' in df_aco.columns:
                            df_aco['EQ_Bujur'] = df_aco['Bujur']
                    else:
                        df_aco = df_train
                except Exception as e:
                    logger.warning(f"[GA] Failed to load ACO CSV for map: {e}")
                    df_aco = df_train

                out_map_path = os.path.join(self.output_dir, "ga_from_aco_map.html")
                self.visualizer.generate_map(
                    best_path=[],
                    df=df_aco,
                    pred_info={
                        "aco_center_lat": df_aco["EQ_Lintang"].mean(),
                        "aco_center_lon": df_aco["EQ_Bujur"].mean() #Rumus 3.15
                    },
                    out_path=out_map_path,
                    ga_vectors=ga_vectors
                )

                logger.info("=== GA ENGINE COMPLETE (ACO Mode) ===".center(80))
                pred_info = {
                    "bearing_degree": pred["bearing_degree"],
                    "distance_km": pred["distance_km"]
                }

                df_train = self._attach_ga_targets(df_train, pred_info)

                return [], {
                    "map": out_map_path,
                    "prediction": {
                        "bearing_degree": pred["bearing_degree"],
                        "distance_km": pred["distance_km"]
                    }
                }

            except Exception as e:
                logger.error(f"[GA] Error processing ACO input: {e}", exc_info=True)
                # Fallback ke standar jika error
        
        # --- FALLBACK LOGIC (Standard GA) ---
        logger.warning("[GA] ACO Input not found or Error. Running Standard GA.")

        # 1. Sanitization
        clean_df = self.sanitizer.execute(df_train)

        # 2. Fitness Engine
        fit_engine = PhysicsFitnessEngine(clean_df, self.cfg.fitness_weights)

        # 3. Evolution
        evo = EvolutionaryController(self.cfg, fit_engine, self.checkpoint_mgr)
        
        # --- FIX 2: Pass 'clean_df' sebagai argumen ---
        # Ini mengatasi error TypeError: missing 1 required positional argument: 'df_train'
        best_idx, log_df, hof = evo.run()

        
        # Clean indices
        if isinstance(best_idx, tuple): best_idx = list(best_idx)
        if len(best_idx) == 1 and isinstance(best_idx[0], tuple): best_idx = list(best_idx[0])
        best_idx = [int(x) for x in best_idx]

        max_idx = len(clean_df) - 1
        safe_idx = [i for i in best_idx if isinstance(i, int) and 0 <= i <= max_idx] 

        if not safe_idx:
            df_opt = clean_df.copy().reset_index(drop=True)
        else:
            df_opt = clean_df.iloc[safe_idx].reset_index(drop=True)

        # 5. Prediction
        pred = fit_engine.predict_next_event(df_opt, n_seg=getattr(self.cfg, "ga_segment_window", 5))
        
        df_train = self._attach_ga_targets(df_train, pred)

        if isinstance(pred, dict) and pred:
            self._write_back_to_aco_json(pred)

        # 6. Visualization
        self.visualizer.generate_map(safe_idx, clean_df, pred, self.map_path)

        # 7. Export
        meta = {
            "Timestamp": datetime.now().isoformat(),
            "PredictedBearing": pred.get("bearing_degree", None),
            "PredictedDistanceKM": pred.get("distance_km", None),
        }
                # Prepare GA input (if any) from existing aco_to_ga.json
        ga_input = None
        try:
            if os.path.exists(aco_json_path):
                with open(aco_json_path, "r", encoding="utf-8") as f:
                    _p = json.load(f)
                ga_input = {
                    "center_lat": _p.get("center_lat"),
                    "center_lon": _p.get("center_lon"),
                    "impact_area_km2": _p.get("impact_area_km2", _p.get("impact_area"))
                }
        except Exception:
            ga_input = None

        # GA output (representative vector)
        ga_output = [{
            "angle_deg": pred.get("bearing_degree"),
            "direction": GeoMathCore.bearing_to_compass_static(pred.get("bearing_degree", 0.0))
        }]

        # Export with GA sheets
        try:
            self.exporter.export(clean_df, df_opt, meta, ga_input=ga_input, ga_output=ga_output)
        except Exception as e:
            logger.warning(f"[GA] Excel export (Fallback Mode) failed: {e}", exc_info=True)

        try:
            log_df.to_csv(self.log_path, index=False)
        except Exception: pass

        logger.info("=== GA ENGINE COMPLETE (Fallback Mode) ===".center(80))

        return best_idx, {"map": self.map_path, "prediction": pred}
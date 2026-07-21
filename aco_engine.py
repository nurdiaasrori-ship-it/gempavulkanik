"""
ACO Engine (Titanium Edition)
Module: Advanced Ant Colony Optimization for Seismic Risk Zoning
"""
import os # operasi path dan file system
import math # operasi matematis
import json # ekspor data ke JSON
import pickle # simpan dan load state ACO (pheromone memory)
import time # timing / delay
import logging # sistem logging terpusat
from typing import List, Optional # type hinting 

import numpy as np # kompytasi numerik dan matrix
import pandas as pd # manipulasi data
import folium # visualisasi peta interaktif
from folium.plugins import HeatMap  # plugin heatmap folium

# --- KONSTANTA ---
R_EARTH_KM = 6371.0 # Jari-jari bumi dalam kilometer -> haversine
EPSILON = 1e-12 # nilai kecil untuk menghindari pembagian nol
DEFAULT_PHEROMONE = 0.1 # nilai awal pheromone
MAX_PHEROMONE = 10.0 # batas maksimum pheromone
MIN_PHEROMONE = 0.01 # batas minimum pheromone

logger = logging.getLogger("ACO_Engine_Master") # logger utama ACO
logger.addHandler(logging.NullHandler()) # cegah error jika tidak ada handler
logger.setLevel(logging.DEBUG) # level debug penuh

# ==========================================
# 1. GEO UTILITIES
# ==========================================
class GeoMath: # kumpulan fungsi geo-matematis
    @staticmethod # haversine vectorized untuk matriks jarak
    def haversine_vectorized(lat_array, lon_array): # menghitung jarak Haversine antar titik
        lat_rad = np.radians(lat_array) # konversi ke derajat ke radian (syarat rumus harversine)
        lon_rad = np.radians(lon_array) # konversi ke derajat ke radian (syarat rumus harversine)
        # hitung selisih lat dan lon antar semua pasangan titik
        dlat = lat_rad[:, np.newaxis] - lat_rad  
        dlon = lon_rad[:, np.newaxis] - lon_rad 
        # hitung jarak haversine ( rumus inti jarak bola bumi)
        a = (
            np.sin(dlat / 2.0) ** 2 +
            np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad) *
            np.sin(dlon / 2.0) ** 2
        ) 
        a = np.clip(a, 0.0, 1.0)  # keamanan numerik (floating error)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))  # Konversi ke jarak permukaan bumi (km)
        dist_matrix = R_EARTH_KM * c # jarak akhir dalam kilometer 
        np.fill_diagonal(dist_matrix, 0.0) # jarak ke diri sendiri = 0
        return dist_matrix + EPSILON  # menambahkan epsilon untuk hindari pembagian nol di heuristic

# ============
# 2. ANT AGENT
# ============
class AntAgent: # representasi satu semut virtual dalam koloni
    def __init__(self, ant_id: int, start_node: int, alpha: float, beta: float, role: str = "Worker"): # inisialisasi properti dasar semut
        self.id = ant_id # ID unik semut
        self.start_node = start_node # node awal semut
        self.current_node = start_node # node saat ini semut
        self.role = role # peran semut (explorer/exploiter)
        self.alpha = alpha # bobot pheromone
        self.beta = beta # bobot heuristic
        self.path = [start_node] # jejak perjalanan semut (jalur yang ditempuh))
        self.visited_mask = None # penanda node yang sudah dikunjungi
        self.accumulated_risk = 0.0 # total risiko yang dikumpulkan semut
    # reset status semut untuk iterasi baru
    def reset(self, new_start_node: int, n_nodes: int): # reset posisi dan status semut
        self.start_node = new_start_node # set node awal baru
        self.current_node = new_start_node # set node saat ini ke awal baru
        self.path = [new_start_node] # reset jejak perjalanan
        self.visited_mask = np.zeros(n_nodes, dtype=bool) # inisialisasi penanda kunjungan
        self.visited_mask[new_start_node] = True # tandai node awal sudah dikunjungi
        self.accumulated_risk = 0.0 # reset risiko terakumulasi
    # gerakan semut ke node berikutnya
    def move_to(self, next_node: int, risk_val: float):
        self.path.append(next_node) # tambahkan node berikutnya ke jejak
        self.visited_mask[next_node] = True # tandai node berikutnya sudah dikunjungi
        self.current_node = next_node # update node saat ini
        self.accumulated_risk += float(max(risk_val, 0.0)) # tambahkan risiko ke total (pastikan non-negatif)

# =======================
# 3. ENVIRONMENT MANAGER
# =======================
class EnvironmentManager: # manajemen lingkungan ACO (matriks jarak, heuristic, pheromone)
    def __init__(self, df: pd.DataFrame, logger_obj): # inisialisasi dengan DataFrame gempa
        self.df = df.copy() # salinan DataFrame input
        self.logger = logger_obj # logger terpusat
        self.n_nodes = len(df) # jumlah node (gempa) dalam DataFrame

        self.dist_matrix = None # matriks jarak antar node
        self.heuristic_matrix = None # matriks heuristic (daya tarik risiko)
        self.pheromone_matrix = None # matriks pheromone (jejak semut)
        
        if self.n_nodes > 0: 
            self._normalize_geo_columns() # normaliasi koordinat
            self._build_distance_matrix() # hitung jarak antar gempa
            self._build_heuristic_matrix() # bangun heuristic berbasis fisika
            self._init_pheromone_matrix() # inisialisasi pheromone

    # normalisasi kolom koordinat agar ACO selalu punya lintang dan bujur
    def _normalize_geo_columns(self):  
        # PRIORITAS KERAS: EQ_Lintang & EQ_Bujur (SAFE)
        if 'EQ_Lintang' in self.df.columns and 'EQ_Bujur' in self.df.columns: 
            self.df['Lintang'] = pd.to_numeric(self.df['EQ_Lintang'], errors='coerce') # konversi kolom EQ_Lintang menjadi numerik lalu simpan ke kolom standar 'Lintang'
            self.df['Bujur'] = pd.to_numeric(self.df['EQ_Bujur'], errors='coerce') # konversi kolom EQ_Bujur menjadi numerik lalu simpan ke kolom standar 'Bujur'
            # Jika hasil konversi menghasilkan NaN (data rusak/kosong)
            if self.df['Lintang'].isna().any() or self.df['Bujur'].isna().any(): # log warning agar mudah terlacak saat debug
                self.logger.warning("[ACO] NaN pada EQ_Lintang/EQ_Bujur -> diisi 0.0") # isi NaN dengan 0.0 agar proses ACO tidak crash
                self.df[['Lintang', 'Bujur']] = self.df[['Lintang', 'Bujur']].fillna(0.0)
            return

        # fallback lama (jaga kompatibilitas)
        col_map_lat = ['Lintang', 'Latitude', 'lat'] # daftar kemungkinan nama kolom lintang (dataset lama/eksternal)
        col_map_lon = ['Bujur', 'Longitude', 'lon'] # daftar kemungkinan nama kolom bujur (dataset lama/eksternal)

        lat_col = next((c for c in col_map_lat if c in self.df.columns), None) # cari kolom lintang yang ada di DataFrame
        lon_col = next((c for c in col_map_lon if c in self.df.columns), None) # cari kolom bujur yang ada di DataFrame
        # jika kolom koordinat tidak ditemukan, lempar error
        if lat_col is None or lon_col is None:
            raise KeyError("[ACO] Kolom koordinat tidak ditemukan")

        self.df['Lintang'] = pd.to_numeric(self.df[lat_col], errors='coerce') # konversi kolom lintang menjadi numerik lalu simpan ke kolom standar 'Lintang'
        self.df['Bujur'] = pd.to_numeric(self.df[lon_col], errors='coerce') # konversi kolom bujur menjadi numerik lalu simpan ke kolom standar 'Bujur'
        # cek jika ada NaN pada kolom koordinat setelah konversi
        if self.df['Lintang'].isna().any() or self.df['Bujur'].isna().any():
            self.logger.warning("[ACO] NaN pada koordinat → diisi 0.0") # log warning agar mudah terlacak saat debug
            self.df[['Lintang','Bujur']] = self.df[['Lintang','Bujur']].fillna(0.0)     # isi NaN dengan 0.0 agar proses ACO tidak crash
    # Membangun Matriks Jarak menggunakan rumus Haversine
    def _build_distance_matrix(self):
        lats = self.df['Lintang'].values # ambil array lintang
        lons = self.df['Bujur'].values # ambil array bujur
        self.dist_matrix = GeoMath.haversine_vectorized(lats, lons) # hitung matriks jarak haversine antar semua titik

    # Membangun Matriks Heuristic berbasis Magnitudo dan Kedalaman
    def _build_heuristic_matrix(self):
        """Membangun Matrix Heuristic untuk ACO berbasis Fisika Gempa."""
        mag_col = next((c for c in ['Magnitudo_Original', 'Magnitudo', 'magnitude', 'mag'] if c in self.df.columns), None)
        depth_col = next((c for c in ['Kedalaman_Original', 'Kedalaman_km', 'Kedalaman (km)', 'depth', 'depth_km'] if c in self.df.columns), None)

        if mag_col is None or depth_col is None:
            # Fallback: jika magnitudo/kedalaman tidak lengkap, gunakan fallback uniform attractiveness
            self.logger.warning("[ACO] Magnitudo/Kedalaman tidak lengkap — memakai fallback uniform heuristic.")
            attractiveness = np.ones(self.n_nodes, dtype=float)
        else:
            mags_raw = pd.to_numeric(self.df[mag_col], errors='coerce').fillna(0.1)
            depths_raw = pd.to_numeric(self.df[depth_col], errors='coerce').fillna(1.0)
            mags = np.clip(mags_raw.values.astype(float), 0.1, None)
            depths = np.clip(depths_raw.values.astype(float), 1.0, None)

            energy_score = np.clip(np.power(mags, 2.5), 1e-4, None)
            depth_factor = np.clip(1.0 / np.power(depths, 0.5), 1e-4, None)
            if np.max(depth_factor) > 0: 
                depth_factor /= np.max(depth_factor)
            attractiveness = energy_score * depth_factor

        attr_matrix = np.tile(attractiveness, (self.n_nodes, 1))
        dist_safe = self.dist_matrix.copy()
        dist_safe[dist_safe < 0.5] = 0.5  # MIN 500 meter
        self.heuristic_matrix = attr_matrix / dist_safe
        # normalisasi matriks heuristic
        np.fill_diagonal(self.heuristic_matrix, 0.0)
        finite_vals = self.heuristic_matrix[np.isfinite(self.heuristic_matrix)]
        max_finite = np.max(finite_vals) if len(finite_vals) > 0 else 1.0
        self.heuristic_matrix[~np.isfinite(self.heuristic_matrix)] = max_finite
        # normalisasi ke [0, 1]
        max_val = np.max(self.heuristic_matrix)
        if max_val <= 1e-9 or np.isnan(max_val): # semua nilai 0 atau NaN
            self.logger.warning("[ACO] Heuristic Matrix kosong/flat/NaN. Menggunakan Fallback uniform.")
            self.heuristic_matrix.fill(1.0)
            np.fill_diagonal(self.heuristic_matrix, 0.0)
        else:
            self.heuristic_matrix /= max_val

        mask_diag = np.eye(self.n_nodes, dtype=bool)
        self.heuristic_matrix[~mask_diag] = np.clip(self.heuristic_matrix[~mask_diag], 1e-4, 1.0)

    # inisialisasi matriks pheromone
    def _init_pheromone_matrix(self):
        self.pheromone_matrix = np.full((self.n_nodes, self.n_nodes), DEFAULT_PHEROMONE, dtype=float) # inisialisasi pheromone konstan
        np.fill_diagonal(self.pheromone_matrix, 0.0) 

    # mendapatkan probabilitas transisi untuk semut di node saat ini
    def get_transition_probabilities(self, current_node: int, ant: AntAgent):
        tau = self.pheromone_matrix[current_node] # ambil pheromone dari node saat ini
        eta = self.heuristic_matrix[current_node] # ambil heuristic dari node saat ini
        prob = np.power(tau, ant.alpha) * np.power(eta, ant.beta) # hitung probabilitas transisi
        # hilangkan node yang sudah dikunjungi
        if ant.visited_mask is not None and len(ant.visited_mask) == len(prob):
            prob = prob.copy()
            prob[ant.visited_mask] = 0.0

        # sanitasi probabilitas: jika semua nol, beri nilai uniform hanya pada non-visited
        if np.sum(prob) <= 0 or not np.isfinite(np.sum(prob)):
            if ant.visited_mask is not None:
                non_visited = ~ant.visited_mask
                if non_visited.any():
                    prob = np.zeros_like(prob, dtype=float)
                    prob[non_visited] = 1.0 / non_visited.sum()
                else:
                    # semua ter-visit: fallback uniform (akan mengijinkan revisit)
                    prob = np.ones_like(prob, dtype=float) / len(prob)
            else:
                prob = np.ones_like(prob, dtype=float) / len(prob)
        return prob

    # terapkan update global pheromone (evaporasi + deposit baru)
    def apply_global_update(self, evaporation_rate: float, deposit_matrix: np.ndarray):
        self.pheromone_matrix *= (1.0 - evaporation_rate)
        self.pheromone_matrix += deposit_matrix
        self.pheromone_matrix = np.clip(self.pheromone_matrix, MIN_PHEROMONE, MAX_PHEROMONE)
        np.fill_diagonal(self.pheromone_matrix, 0.0)
    # reset pheromone dengan smoothing (mengurangi ekstrem)
    def reset_pheromone_smooth(self):
        avg = float(np.mean(self.pheromone_matrix))
        self.pheromone_matrix = 0.5 * self.pheromone_matrix + 0.5 * avg
        np.fill_diagonal(self.pheromone_matrix, 0.0)

# ================
# 4. MAIN ENGINE
# ================
class DynamicAcoEngine: # mesin utama ACO dengan konfigurasi dinamis
    def __init__(self, config): 
        self.logger = logging.getLogger("ACO_Engine_Master")
        self.aco_cfg = self._prepare_config(config)
        self._load_parameters()

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output')) # path output dasar
        self.output_paths = {
            'aco_zoning_excel': os.path.join(base_path, 'aco_results/aco_zoning_data_for_lstm.xlsx'),
            'aco_epicenters_csv': os.path.join(base_path, 'aco_results/aco_epicenters.csv'),
            'aco_state_file': os.path.join(base_path, 'aco_results/aco_brain_state.pkl'),
            'aco_impact_html': os.path.join(base_path, 'aco_results/aco_impact_zones.html'),
            'aco_presentation_excel': os.path.join(base_path, 'aco_results/aco_presentation.xlsx')
        } # pastikan direktori output ada
        os.makedirs(os.path.dirname(self.output_paths['aco_zoning_excel']), exist_ok=True) # buat direktori jika belum ada

        self.env_manager: Optional[EnvironmentManager] = None # manajer lingkungan ACO
        self.colony: List[AntAgent] = [] # koloni semut
        self.best_global_score = -np.inf # skor global terbaik
        self.stagnation_counter = 0 # counter stagnasi
    # siapkan konfigurasi ACO dari dict atau object
    def _prepare_config(self, config):
        if isinstance(config, dict):
            return config
        elif hasattr(config, "__dict__"):
            return {k: v for k, v in vars(config).items() if not k.startswith('__')}
        else:
            self.logger.warning("[ACO] Config tidak dikenali, memakai config kosong.")
            return {}
    # muat parameter ACO dari konfigurasi
    def _load_parameters(self):
        self.n_ants = int(self.aco_cfg.get('n_ants', 50))
        self.n_iterations = int(self.aco_cfg.get('n_iterations', 100))
        self.n_steps = int(self.aco_cfg.get('n_epicenters', 20))
        self.alpha_base = float(self.aco_cfg.get('alpha', 1.0))
        self.beta_base = float(self.aco_cfg.get('beta', 2.0))
        self.rho_base = float(self.aco_cfg.get('evaporation_rate', 0.1))
        self.Q = float(self.aco_cfg.get('pheromone_deposit', 100.0))
        self.risk_threshold = float(self.aco_cfg.get('risk_threshold', 0.7))

    
    

    # inisialisasi koloni semut
    def _initialize_colony(self, n_nodes: int):
        self.colony = []

        if n_nodes <= 0: # jika tidak ada node, tidak perlu inisialisasi
            return
        # tentukan start node berdasarkan heuristic (semakin berisiko, semakin mungkin jadi start)
        start_scores = self.env_manager.heuristic_matrix.sum(axis=1)
        # sanitasi skor start (hapus NaN/Negatif)
        start_scores = np.clip(start_scores, 0.0, None) # pastikan non-negatif
        total = float(start_scores.sum()) # total skor start
        if total <= 0 or not np.isfinite(total): # fallback uniform jika semua skor 0 atau NaN
            start_probs = np.ones(n_nodes, dtype=float) / max(n_nodes, 1)
        else:
            start_probs = start_scores / total
        # buat semut dengan peran berbeda (explorer/exploiter)
        for i in range(self.n_ants):
            if i < int(self.n_ants * 0.2): # 20% semut sebagai explorer
                role = "Explorer" # peran semut
                alpha = self.alpha_base * 0.5 # bobot pheromone lebih rendah
                beta = self.beta_base * 1.5 # bobot heuristic lebih tinggi
            else: # 80% semut sebagai exploiter
                role = "Exploiter" # peran semut
                alpha = self.alpha_base * 1.2 # bobot pheromone lebih tinggi
                beta = self.beta_base * 0.9 # bobot heuristic lebih rendah
            # pilih start node berdasarkan probabilitas
            start_node = int(np.random.choice(np.arange(n_nodes), p=start_probs))
            ant = AntAgent(i, start_node, alpha, beta, role) # buat semut baru
            ant.reset(start_node, n_nodes)  # reset status semut
            self.colony.append(ant) # tambahkan semut ke koloni

    # step semut dalam satu iterasi
    def _step_ants(self):
        active_ants = 0
        for ant in self.colony: # loop setiap semut
            if ant.current_node == -1: 
                continue
            # dapatkan probabilitas transisi dari node saat ini
            probs = self.env_manager.get_transition_probabilities(ant.current_node, ant)
            total = float(np.sum(probs)) # total probabilitas
            # jika tidak ada kemungkinan transisi, semut berhenti
            if total <= 0 or not np.isfinite(total):
                ant.current_node = -1
                continue
            # pilih node berikutnya berdasarkan probabilitas
            norm = probs / total
            next_node = int(np.random.choice(np.arange(self.env_manager.n_nodes), p=norm))
            # dapatkan nilai risiko dari heuristic matrix
            risk_val = self.env_manager.heuristic_matrix[ant.current_node, next_node]
            ant.move_to(next_node, risk_val)
            active_ants += 1

        return active_ants # kembalikan jumlah semut yang masih aktif

    # manajemen siklus hidup koloni per iterasi
    def _manage_lifecycle(self, iteration: int):
        """
        Mengelola siklus hidup koloni per iterasi:
        1. Menghitung deposit pheromone berdasarkan performa semut.
        2. Evaporasi & Update Global Pheromone.
        3. Mendeteksi Stagnasi (Convergence trap).
        4. Re-spawn (Reset posisi) semut untuk iterasi berikutnya.
        """
        # Matriks Delta Pheromone (Akan dijumlahkan ke pheromone utama)
        deposit_matrix = np.zeros_like(self.env_manager.pheromone_matrix, dtype=float)
        iter_best = -np.inf
        
        # 1. LOOP ANALISIS PERFORMANCE SEMUT
        for ant in self.colony:
            score = float(ant.accumulated_risk)
            
            # Abaikan solusi tidak valid (NaN atau <= 0)
            if not np.isfinite(score) or score <= 0:
                continue
            
            # Track best score local di iterasi ini
            if score > iter_best:
                iter_best = score
                
            # Hitung Kualitas Solusi untuk deposit
            # Quality = Total Risk / (Panjang Path).
            # Logika: Risiko tinggi yang dicapai dalam langkah sedikit = Lebih Efisien.
            path_len = len(ant.path)
            ant_quality = score / (path_len + 1e-6)
            
            if ant_quality > 0:
                # Faktor Q mempengaruhi seberapa kuat jejak ditinggalkan
                dep = self.Q * ant_quality 
                
                # Deposit pheromone di sepanjang edge yang dilalui
                path = ant.path
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    # Update undirected (dua arah)
                    deposit_matrix[u, v] += dep
                    deposit_matrix[v, u] += dep

        # 2. ADAPTIVE EVAPORATION (DYNAMIC RHO)
        # Jika stagnasi tinggi, tingkatkan evaporasi agar jejak lama cepat hilang (exploration mode)
        current_rho = self.rho_base
        if self.stagnation_counter > 5:
            # Batas rho max 0.9 agar tidak menghapus total
            current_rho = min(0.9, self.rho_base * 1.5)
            
        # Terapkan update global ke Environment (Evaporasi + Deposit baru)
        self.env_manager.apply_global_update(current_rho, deposit_matrix)

        # 3. STAGNATION CHECK
        if iter_best > self.best_global_score:
            self.best_global_score = iter_best
            self.stagnation_counter = 0 # Reset counter jika ada kemajuan
        else:
            self.stagnation_counter += 1
            
        # Mekanisme Anti-Stagnasi: Soft Reset Pheromone
        if self.stagnation_counter > (self.n_iterations * 0.2):
            self.logger.info(f"[ACO Iter-{iteration}] Stagnasi ({self.stagnation_counter}) → Smooth Reset Pheromone.")
            self.env_manager.reset_pheromone_smooth()
            self.stagnation_counter = 0

        # 4. RE-SPAWN ANTS (POSITION RESET)
        n_nodes = self.env_manager.n_nodes
        if n_nodes <= 0:
            return

        # Tentukan start node berikutnya berdasarkan heuristic (Semakin bahaya node, semakin mungkin jadi start)
        # Sum axis=1: Total attractiveness node tersebut
        if hasattr(self.env_manager, 'heuristic_matrix'):
            start_scores = np.sum(self.env_manager.heuristic_matrix, axis=1)
        else:
            start_scores = np.ones(n_nodes)
            
        # Sanitasi Score (hapus NaN/Negatif)
        start_scores = np.nan_to_num(start_scores, nan=0.0)
        start_scores = np.clip(start_scores, 0.0, None)
        
        total_score = float(start_scores.sum())
        
        # Buat Probabilitas Start
        if total_score <= 1e-9:
            # Fallback: Uniform Probability jika semua 0
            start_probs = np.ones(n_nodes, dtype=float) / n_nodes
        else:
            start_probs = start_scores / total_score
            # [FIX CRITICAL]: Re-normalize sum agar PERSIS 1.0 (numpy choice rewel soal ini)
            start_probs /= start_probs.sum()

        # Reset posisi setiap semut untuk iterasi depan
        for ant in self.colony:
            try:
                new_start = np.random.choice(np.arange(n_nodes), p=start_probs)
            except ValueError:
                # Jika masih error floating point sum != 1, fallback random uniform
                new_start = np.random.randint(0, n_nodes)
                
            ant.reset(int(new_start), n_nodes)
    # Simpan hasil ACO ke disk (Excel & CSV)
    def _compute_impact_center(self, df):
        lat_col = None # kolom lintang
        lon_col = None # kolom bujur
        # deteksi kolom koordinat
        for c in df.columns:
            c_low = c.lower() # nama kolom dalam huruf kecil
            if c_low in ['lintang', 'latitude', 'lat']: # deteksi kolom lintang
                lat_col = c
            if c_low in ['bujur', 'longitude', 'lon']: # deteksi kolom bujur
                lon_col = c
        # jika kolom koordinat tidak ditemukan, lempar error
        if lat_col is None or lon_col is None:
            raise KeyError(f"[ACO] Kolom koordinat tidak ditemukan. Kolom tersedia: {list(df.columns)}")

        # Simpan koordinat sebagai numerik
        coords = df[[lat_col, lon_col]].copy()
        coords[lat_col] = pd.to_numeric(coords[lat_col], errors='coerce')
        coords[lon_col] = pd.to_numeric(coords[lon_col], errors='coerce')

        weights = pd.to_numeric(df.get('PheromoneScore', pd.Series(0, index=df.index)), errors='coerce').fillna(0.0).values
        valid_mask = coords[lat_col].notna() & coords[lon_col].notna()
        # jika tidak ada koordinat valid, gunakan mean sederhana
        if valid_mask.sum() == 0:
            # gunakan mean sederhana
            lat_center = float(coords[lat_col].mean(skipna=True) or 0.0)
            lon_center = float(coords[lon_col].mean(skipna=True) or 0.0)
        else:
            # gunakan rata-rata bobot pheromone
            w = weights[valid_mask]
            # jika semua bobot nol, fallback ke mean sederhana
            if np.sum(w) <= 1e-9:
                lat_center = float(coords.loc[valid_mask, lat_col].mean())
                lon_center = float(coords.loc[valid_mask, lon_col].mean())
            else:
                lat_center = float(np.average(coords.loc[valid_mask, lat_col].values, weights=w))
                lon_center = float(np.average(coords.loc[valid_mask, lon_col].values, weights=w))

        return {
            "center_lat": lat_center,
            "center_lon": lon_center,
            "lat_column_used": lat_col,
            "lon_column_used": lon_col
        } # kembalikan info pusat dampak
    # hitung luas area terdampak berdasarkan radius visual
    def _compute_impact_area(self, df):
        """
        Hitung estimasi luas area terdampak (km²)
        Berdasarkan radius visual hasil ACO
        """
        # Pastikan kolom radius visual ada
        if df.empty or 'Radius_Visual_KM' not in df.columns:
            self.logger.warning("[ACO] Tidak dapat menghitung impact area (data kosong / kolom hilang)")
            return {"impact_area_km2": 0.0}
        # Konversi radius ke numerik
        radius_km = pd.to_numeric(df['Radius_Visual_KM'], errors='coerce').fillna(0.0)

        # Area lingkaran: πr²
        areas = math.pi * np.power(radius_km.values, 2)

        # Total area unik (bukan sum mentah → konservatif)
        impact_area = float(np.nanmax(areas))

        impact_area_kecil = impact_area / 10.0

        return {
            "impact_area_km2": round(impact_area_kecil, 3)
        }

    def _export_for_ga(self, df, center_info):
        """
        Export minimal ACO -> GA payload.
        Client requirement: GA input HANYA center & impact area.
        """
        area_info = self._compute_impact_area(df)
        ga_input = {
            "center_lat": float(center_info.get("center_lat", 0.0)),
            "center_lon": float(center_info.get("center_lon", 0.0)),
            "impact_area_km2": float(area_info.get("impact_area_km2", 0.0))
        }

        # Save to aco_results/aco_to_ga.json (consistent with GA expecting this path)
        output_dir = os.path.dirname(self.output_paths['aco_state_file'])
        ga_path = os.path.join(output_dir, "aco_to_ga.json")
        from tempfile import NamedTemporaryFile
        try:
            dirn = os.path.dirname(ga_path)
            with NamedTemporaryFile('w', dir=dirn, delete=False, encoding='utf-8') as tf:
                existing = {}
                if os.path.exists(ga_path):
                    try:
                        with open(ga_path, 'r', encoding='utf-8') as f:
                            existing = json.load(f)
                    except Exception:
                        existing = {}

                existing.update(ga_input)  # hanya update center & area
                json.dump(existing, tf, indent=2, ensure_ascii=False)
                tmp = tf.name
            os.replace(tmp, ga_path)  # atomic move
            self.logger.info(f"[ACO] Minimal GA input saved → {ga_path} (atomic)")
        except Exception as e:
            self.logger.error(f"[ACO] Failed to write ACO->GA json: {e}", exc_info=True)
        return ga_input



    # menjalankan ACO pada DataFrame input
    def run(self, df: pd.DataFrame):
        seed = self.aco_cfg.get('random_seed', None)
        if seed is not None:
            np.random.seed(seed)

        self.logger.debug(f"[ACO] DataFrame masuk: {df.shape[0]} baris, {df.shape[1]} kolom")
        self.logger.debug(f"[ACO] Kolom: {list(df.columns)}")
        self.logger.debug(df.head(5).to_string())
        if 'EQ_Lintang' in df.columns and 'EQ_Bujur' in df.columns: # cek kolom koordinat
            print("[DEBUG ACO] Kolom koordinat tersedia")
            print(df[['EQ_Lintang', 'EQ_Bujur']].tail(10))
            print("NaN count:", df[['EQ_Lintang', 'EQ_Bujur']].isna().sum())
        else:
            print("[DEBUG ACO] Kolom koordinat EQ_Lintang/EQ_Bujur tidak ditemukan!")

        if df is None or df.empty:
            self.logger.warning("[ACO] DataFrame kosong.")
            print("DEBUG: VRP DF KOSONG, ACO Center = nan")
            return df, {}

        # inisialisasi environment manager
        self.env_manager = EnvironmentManager(df, self.logger)
        # khusus live event dengan 1 node
        if self.env_manager.n_nodes <= 1:
            if os.path.exists(self.output_paths['aco_state_file']):
                try: # coba load state lama
                    with open(self.output_paths['aco_state_file'], 'rb') as f:
                        state = pickle.load(f)
                    
                    # Logika ini HANYA JALAN PADA SINGLE EVENT
                    if 'PheromoneScore' in df.columns:
                         # Ambil Skor PheromoneScore yang sudah dihitung di FE sebelumnya
                         # Jika ACO dijalankan, skornya pasti sudah ada
                         final_score = df['PheromoneScore'].values 
                         return df, {"pheromone_matrix": state.get('pheromone_matrix')}
                # Jika ada state, gunakan untuk hitung skor risiko        
                except Exception as e:
                    self.logger.warning(f"[ACO] Gagal memproses live event dengan state: {e}. Mengembalikan skor 0.")
                    df['PheromoneScore'] = 1e-4 # Fallback minimal
                    df['Pheromone_Score'] = 1e-4
                    df['Risk_Index'] = 0.01
                    return df, {}
            else:
                 # Jika tidak ada state, tidak ada ACO.
                 df['PheromoneScore'] = 1e-4
                 df['Pheromone_Score'] = 1e-4
                 df['Risk_Index'] = 0.01
                 self.logger.warning("[ACO] Live Mode: State tidak ditemukan. Menggunakan skor risiko minimum.")
                 return df, {}

        # coba load state lama
        if os.path.exists(self.output_paths['aco_state_file']):
            try:
                with open(self.output_paths['aco_state_file'], 'rb') as f:
                    state = pickle.load(f)
                old_matrix = state.get('pheromone_matrix')
                if isinstance(old_matrix, np.ndarray) and old_matrix.shape == self.env_manager.pheromone_matrix.shape:
                    self.logger.info("[ACO] Melanjutkan dari brain state sebelumnya.")
                    self.env_manager.pheromone_matrix = old_matrix
            except Exception as e:
                self.logger.warning(f"[ACO] Gagal load state lama: {e}")
        # inisialisasi koloni semut
        self._initialize_colony(self.env_manager.n_nodes)
        # loop utama ACO
        for it in range(self.n_iterations):
            for _ in range(max(self.n_steps - 1, 1)):
                if self._step_ants() == 0:
                    break
            self._manage_lifecycle(it)

        # simpan brain state
        try:
            with open(self.output_paths['aco_state_file'], 'wb') as f:
                pickle.dump({'pheromone_matrix': self.env_manager.pheromone_matrix}, f)
        except Exception as e:
            self.logger.warning(f"[ACO] Gagal simpan brain state: {e}")
        # finalisasi hasil
        df_out, meta = self._finalize_results(df)
        center_info = self._compute_impact_center(df_out)
        # =====================================================
        # [OPSI 1] TIME-AWARE ACO EVENT (AFTER ACO FINISHED)
        # =====================================================

        # =========================
        # EXTRACT EVENT TIMESTAMP
        # =========================
        event_time = None
        time_candidates = [
            'Tanggal', 'Acquired_Date', 'Timestamp', 'EventTime',
            'Tanggal_Kejadian', 'Date', 'date'
        ]

        for c in time_candidates:
            if c in df_out.columns:
                ts_series = pd.to_datetime(df_out[c], errors='coerce').dropna()
                if not ts_series.empty:
                    event_time = ts_series.iloc[-1]
                    break

        if event_time is None or pd.isna(event_time):
            event_time = pd.Timestamp.utcnow()

        # =========================
        # BUILD ACO EVENT RECORD
        # =========================
        area_info = self._compute_impact_area(df_out)

        aco_event = {
            "timestamp": event_time,
            "aco_center_lat": float(center_info.get("center_lat", 0.0)),
            "aco_center_lon": float(center_info.get("center_lon", 0.0)),
            "aco_area_km2": float(area_info.get("impact_area_km2", 0.0))
        }

        # =========================
        # APPEND ACO EVENT (CSV)
        # =========================
        try:
            aco_event_path = os.path.join(
                os.path.dirname(self.output_paths['aco_state_file']),
                "aco_events.csv"
            )

            df_event = pd.DataFrame([aco_event])

            if os.path.exists(aco_event_path):
                df_old = pd.read_csv(aco_event_path, parse_dates=['timestamp'])
                df_all = pd.concat([df_old, df_event], ignore_index=True)
            else:
                df_all = df_event

            df_all.sort_values("timestamp", inplace=True)
            df_all.to_csv(aco_event_path, index=False)

            self.logger.info(f"[ACO] Event ACO tersimpan → {aco_event_path}")

        except Exception as e:
            self.logger.warning(f"[ACO] Gagal menyimpan ACO event: {e}")

        # Export presentation Excel (untuk klien/presentasi)
        try:
            self._export_presentation_excel(df_out, center_info, meta)
        except Exception:
            # jangan gagalkan pipeline jika export gagal
            self.logger.warning("[ACO] _export_presentation_excel gagal, melanjutkan pipeline.")

        # Validasi center
        c_lat = center_info.get('center_lat', None)
        c_lon = center_info.get('center_lon', None)
        if c_lat is None or c_lon is None or (abs(c_lat) < 1e-6 and abs(c_lon) < 1e-6):
            self.logger.warning("[ACO] Center tidak valid/terlalu kecil → tidak menulis ACO->GA JSON.")
        else:
            self._export_for_ga(df_out, center_info)
        self._save_to_disk(df_out)
        self._generate_visuals(df_out)
        return df_out, meta

    # finalisasi hasil ACO
    def _finalize_results(self, df: pd.DataFrame):
        """
        Hitung skor risiko node dari matriks pheromone,
        skala ke [1e-4, 1] dan juga ke indeks 0–100.
        """
        if self.env_manager is None or self.env_manager.pheromone_matrix is None:
            # Fallback jika matrix belum tersedia
            n_nodes = len(df)
            norm_scores = np.ones(n_nodes, dtype=float) * 1e-4
        else:
            node_importance = np.sum(self.env_manager.pheromone_matrix, axis=0)
            # quantile untuk buang outlier ekstrem
            q01, q99 = np.quantile(node_importance, [0.01, 0.99])
            q01 = q01 if np.isfinite(q01) else float(node_importance.min())
            q99 = q99 if np.isfinite(q99) else float(node_importance.max())
            # normalisasi ke [0, 1] dengan clipping
            if q99 <= q01 + 1e-9:
                norm_scores = np.ones_like(node_importance, dtype=float) * 0.5
            else:
                clipped = np.clip(node_importance, q01, q99)
                norm_scores = (clipped - q01) / (q99 - q01)
            
            norm_scores = np.clip(norm_scores, 1e-4, 1.0)
            norm_scores = np.nan_to_num(norm_scores, nan=1e-4)

            if np.sum(norm_scores) <= 1e-6:
                self.logger.warning("[ACO] All pheromone scores zero → fallback uniform")
                norm_scores[:] = 1.0 / len(norm_scores)
            
        # Siapkan DataFrame output
        df_out = self.env_manager.df.copy() if self.env_manager else df.copy()
        df_out['PheromoneScore'] = norm_scores
        df_out['Pheromone_Score'] = norm_scores
        df_out['Risk_Index'] = (norm_scores * 100.0).round(2)

        # Tentukan Status Zona secara pasti
        df_out['Status_Zona'] = df_out['PheromoneScore'].apply(
            lambda x: 'Terdampak' if x >= self.risk_threshold else 'Aman'
        )

        # Fallback radius minimal jika Magnitudo hilang
        mag_col = 'Magnitudo_Original' if 'Magnitudo_Original' in df_out.columns else 'Magnitudo'
        mags = np.clip(df_out[mag_col].values if mag_col in df_out.columns else np.ones(len(df_out)), 0.0, None)
        pher = df_out['Pheromone_Score']
        base_r = np.power(mags, 1.3)
        radius_km = base_r * (1.0 + 0.5 * pher)
        radius_km = np.clip(radius_km, 3.0, 80.0)
        df_out['Radius_Visual_KM'] = radius_km
        # Pastikan kolom koordinat ada dan numerik
        for src, dst in [('EQ_Lintang','Lintang'), ('EQ_Bujur','Bujur')]:
            if dst not in df_out.columns or df_out[dst].isna().all():
                if src in df_out.columns:
                    df_out[dst] = pd.to_numeric(df_out[src], errors='coerce')
        
        df_out[['Lintang','Bujur']] = df_out[['Lintang','Bujur']].fillna(0.0)

        return df_out, {
            "pheromone_matrix": self.env_manager.pheromone_matrix if self.env_manager else None
        } # kembalikan DataFrame dan metadata

    # ===========
    # SAVE FILES
    # ===========
    # simpan hasil ACO ke disk
    def _save_to_disk(self, df):
        mag_col = 'Magnitudo_Original' if 'Magnitudo_Original' in df.columns else 'Magnitudo'
        depth_col = 'Kedalaman_Original' if 'Kedalaman_Original' in df.columns else 'Kedalaman_km'

        cols = [
            'Tanggal', 'Lintang', 'Bujur', mag_col, depth_col, 'Lokasi',
            'PheromoneScore', 'Risk_Index', 'Status_Zona', 'Radius_Visual_KM'
        ] # kolom output penting
        
        # Tambahkan kolom yang mungkin sudah di-rename oleh FeatureEngineer
        final_df = df.copy()
        if 'EQ_Lintang' in df.columns and 'Lintang' not in df.columns:
            final_df['Lintang'] = df['EQ_Lintang']
        if 'EQ_Bujur' in df.columns and 'Bujur' not in df.columns:
            final_df['Bujur'] = df['EQ_Bujur']
        if 'Nama' in df.columns and 'Lokasi' not in df.columns:
            final_df['Lokasi'] = df['Nama']
        if 'Acquired_Date' in df.columns and 'Tanggal' not in df.columns:
            final_df['Tanggal'] = df['Acquired_Date']
        if 'Lintang' not in final_df.columns or 'Bujur' not in final_df.columns:
            self.logger.warning("[ACO] final_df belum punya Lintang/Bujur sebelum save; menambahkan default 0.0")
            final_df['Lintang'] = pd.to_numeric(final_df.get('Lintang', 0.0), errors='coerce').fillna(0.0)
            final_df['Bujur'] = pd.to_numeric(final_df.get('Bujur', 0.0), errors='coerce').fillna(0.0)
        # susun ulang dan ganti nama kolom untuk output
        final_df = final_df[[c for c in cols if c in final_df.columns]].rename(columns={mag_col: 'Magnitudo', depth_col: 'Kedalaman'})
        # simpan ke Excel dan CSV
        try:
            final_df.to_excel(self.output_paths['aco_zoning_excel'], index=False)
            final_df.to_csv(self.output_paths['aco_epicenters_csv'], index=False)
        except Exception as e:
            self.logger.error(f"Gagal menyimpan output ACO: {e}")


    def _export_presentation_excel(self, df_out: pd.DataFrame, center_info: dict, meta: dict):
        """
        Export presentation-ready Excel with multiple sheets:
        - Parameters
        - Node_Summary
        - Pheromone_Stats
        - Center_and_Area
        - Detailed_Epicenters
        """
        path = self.output_paths.get('aco_presentation_excel')
        if not path:
            self.logger.warning("[ACO] Path presentation excel tidak diset.")
            return

        try:
            # =========================
            # Parameters sheet
            # =========================
            params_df = pd.DataFrame(
                list(self.aco_cfg.items()),
                columns=['Parameter', 'Value']
            )

            # =========================
            # Node summary sheet
            # =========================
            summary_cols = [
                c for c in [
                    'Tanggal', 'Lintang', 'Bujur', 'Magnitudo', 'Kedalaman', 'Lokasi',
                    'PheromoneScore', 'Risk_Index', 'Status_Zona', 'Radius_Visual_KM'
                ] if c in df_out.columns
            ]
            node_summary = df_out[summary_cols].copy() if summary_cols else pd.DataFrame()

            # =========================
            # Pheromone statistics
            # =========================
            pher_stats = {'info': 'pheromone matrix not available'}
            if self.env_manager is not None and hasattr(self.env_manager, 'pheromone_matrix'):
                pher_mat = self.env_manager.pheromone_matrix
                if pher_mat is not None:
                    pher_stats = {
                        'min': float(np.min(pher_mat)),
                        'max': float(np.max(pher_mat)),
                        'mean': float(np.mean(pher_mat)),
                        'median': float(np.median(pher_mat)),
                        'std': float(np.std(pher_mat)),
                        'shape': str(pher_mat.shape)
                    }

            pher_stats_df = pd.DataFrame(
                list(pher_stats.items()),
                columns=['Metric', 'Value']
            )

            # =========================
            # Center & impact area
            # =========================
            area_info = self._compute_impact_area(df_out)
            center_and_area = {
                'center_lat': center_info.get('center_lat'),
                'center_lon': center_info.get('center_lon'),
                'impact_area_km2': area_info.get('impact_area_km2')
            }
            center_df = pd.DataFrame(
                list(center_and_area.items()),
                columns=['Metric', 'Value']
            )

            # =========================
            # WRITE EXCEL (FIXED)
            # =========================
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
                pher_stats_df.to_excel(writer, sheet_name='Pheromone_Stats', index=False)
                center_df.to_excel(writer, sheet_name='Center_and_Area', index=False)

                if not node_summary.empty:
                    node_summary.to_excel(writer, sheet_name='Node_Summary', index=False)

                df_out.to_excel(writer, sheet_name='Detailed_Epicenters', index=False)

            self.logger.info(f"[ACO] Presentation Excel tersimpan → {path}")

        except Exception as e:
            self.logger.error(
                f"[ACO] Gagal membuat presentation excel: {e}",
                exc_info=True
            )
            raise

    # ==========================================
    # VISUAL: CIRCLE KUNING + POPUP LENGKAP
    # ==========================================
    # buat visualisasi peta dampak ACO
    def _generate_visuals(self, df):
        """
        Visual:
        - Circle orange (zona dampak) + titik pusat merah
        - Popup (TANPA TANGGAL):
          lokasi estimasi (lat, lon), magnitudo, kedalaman,
          radius prediksi, Risk Score, Risk Index
        """
        if df.empty or 'Lintang' not in df.columns or 'Bujur' not in df.columns:
            self.logger.warning("[ACO] Dataframe kosong atau kurang kolom koordinat untuk visualisasi.")
            return

        try:
            df = df.copy()

            # =========================
            # NORMALISASI KOORDINAT
            # =========================
            df['Lintang'] = pd.to_numeric(df['Lintang'], errors='coerce')
            df['Bujur'] = pd.to_numeric(df['Bujur'], errors='coerce')
            df = df.dropna(subset=['Lintang', 'Bujur'])
            if df.empty:
                self.logger.warning("[ACO] Semua baris koordinat NaN. Visualisasi dibatalkan.")
                return

            # =========================
            # NORMALISASI KOLOM TANGGAL (robust)
            # =========================
            date_candidates = [
                'Tanggal', 'Acquired_Date', 'AcquiredDate', 'Acquired Date',
                'Acquired_DateTime', 'Acquired_Time', 'Acquired_Timestamp',
                'EventTime', 'Time_UTC', 'Timestamp', 'Date', 'date', 'Tanggal_Kejadian'
            ]
            date_col = next((c for c in date_candidates if c in df.columns), None)
            # fungsi parsing tanggal robust
            def parse_date_series(s):
                # coba langsung -> pandas
                s_parsed = pd.to_datetime(s, errors='coerce', utc=True)
                # jika banyak NaT, mungkin kolom berisi epoch integer (deteksi)
                if s_parsed.isna().sum() > 0 and s.dropna().dtype.kind in ('i','u','f'):
                    # coba sebagai epoch seconds lalu milliseconds
                    s_epoch_s = pd.to_datetime(s.dropna().astype('int64'), unit='s', errors='coerce', utc=True)
                    if s_epoch_s.notna().sum() > 0:
                        # gabungkan hasil (isi yang ter-parse)
                        s_parsed = s_parsed.combine_first(s_epoch_s.reindex(s.index))
                    else:
                        s_epoch_ms = pd.to_datetime(s.dropna().astype('int64'), unit='ms', errors='coerce', utc=True)
                        s_parsed = s_parsed.combine_first(s_epoch_ms.reindex(s.index))
                return s_parsed

            if date_col:
                df['__Tanggal_parsed__'] = parse_date_series(df[date_col])
            else:
                # kalau tidak ada kolom tanggal, coba beberapa kolom lain (nama sebagian)
                found = None
                for c in df.columns:
                    if any(token in c.lower() for token in ['date','time','timestamp','acquired']):
                        found = c
                        break
                if found:
                    df['__Tanggal_parsed__'] = parse_date_series(df[found])
                else:
                    df['__Tanggal_parsed__'] = pd.NaT

            # format string yang ramah (jika datetime valid), else '-'
            df['__Tanggal__'] = df['__Tanggal_parsed__'].dt.tz_convert(None).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['__Tanggal__'] = df['__Tanggal__'].fillna('-')

            # =========================
            # LOKASI ESTIMASI (OFFLINE)
            # =========================
            df['Lokasi'] = df.apply(
                lambda r: f"Lat {r['Lintang']:.2f}, Lon {r['Bujur']:.2f}",
                axis=1
            )

            # =========================
            # INIT MAP
            # =========================
            center_lat = float(df['Lintang'].mean())
            center_lon = float(df['Bujur'].mean())

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles="OpenStreetMap"
            )

            # =========================
            # LOOP VISUAL
            # =========================
            for _, r in df.iterrows():
                rad_km = float(r.get('Radius_Visual_KM', 0.0))
                radius_m = rad_km * 1000.0

                mag = r.get('Magnitudo', r.get('Magnitudo_Original', None))
                try:
                    mag = f"{float(mag):.1f}"
                except Exception:
                    mag = "-"

                depth = r.get('Kedalaman', r.get('Kedalaman (km)', r.get('Kedalaman_km', None)))
                try:
                    depth = f"{float(depth):.0f}"
                except Exception:
                    depth = "-"

                risk = float(r.get('PheromoneScore', r.get('Pheromone_Score', 0.0)))
                risk_idx = float(r.get('Risk_Index', risk * 100.0))

                # ambil tanggal yang sudah diformat
                tanggal_str = r.get('__Tanggal__', '-') if pd.notna(r.get('__Tanggal__', None)) else '-'

                popup_html = f"""
                <b>Tanggal Kejadian:</b> {tanggal_str}<br>
                <b>Lokasi (Estimasi):</b> {r['Lokasi']}<br>
                <b>Magnitudo:</b> {mag}<br>
                <b>Kedalaman:</b> {depth} km<br>
                <b>Radius Prediksi:</b> {rad_km:.2f} km<br>
                <b>Risk Score (0–1):</b> {risk:.4f}<br>
                <b>Risk Index (0–100):</b> {risk_idx:.2f}
                """

                popup = folium.Popup(popup_html, max_width=320)

                folium.Circle(
                    location=[r['Lintang'], r['Bujur']],
                    radius=radius_m,
                    color='orange',
                    fill=True,
                    fill_opacity=0.25,
                    weight=1.5,
                    popup=popup
                ).add_to(m)

                folium.CircleMarker(
                    location=[r['Lintang'], r['Bujur']],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_opacity=1.0
                ).add_to(m)

            m.save(self.output_paths['aco_impact_html'])
            self.logger.info(f"[ACO] Visual ACO tersimpan → {self.output_paths['aco_impact_html']}")

        except Exception as e:
            self.logger.error(f"[ACO] Gagal membuat visual ACO: {e}", exc_info=True)
# -*- coding:utf-8 -*-
# filename: datasets.py
# 这个版本的 datasets.py 实现了以下核心功能:
# 1. create_loaders: 完整实现了双归一化器（雷达+地面）的离线拟合与加载。
# 2. SeasonalRadarDataset: 在 __getitem__ 中实现了地面数据的降采样。
# 3. 数据分流: 清晰地将雷达和地面数据作为两个独立的数据流进行处理和返回。

import os
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from normalize import Normalizer # 引入通用的 Normalizer 类
from collections import defaultdict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 缓存管理器---
class FileCacheManager:
    _instance = None
    _cache = {}
    _max_size = 200 # 根据硬件（内存）情况设置
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def get_file(self, file_path):
        if file_path in self._cache:
            return self._cache[file_path]
        df = pd.read_parquet(file_path)
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[file_path] = df
        return df
file_cache = FileCacheManager()
# 使用文件缓存管理器读取Parquet文件
def _read_parquet_cached(file_path: str) -> pd.DataFrame:
    return file_cache.get_file(file_path)
# 计算样本难度分
def _compute_file_difficulty(file_path, config):
    """
    计算单个文件的难度因子。
    修订内容：集中参数管理，输出各因子详细得分。
    """
    df = pq.read_table(file_path).to_pandas()
    dataset_cfg = config
    valid_ranges = dataset_cfg['valid_ranges']
    height = dataset_cfg['height']
    weights = dataset_cfg.get('difficulty_weights', {})
    # --- 1. 参数集中配置中心 ---
    diff_params = {
        'window_size': 120,  # 滑动窗口大小 (分钟)
        'overlap_h_limit': 40.0,  # 垂直重叠敏感高度阈值 (Bins)
        'overlap_step': 60,  # 重叠计算步长 (分钟)
        'near_ground_h': 50,  # 近地面定义高度 (Bins)
        'trans_k': 0.8,  # 时域转换 Sigmoid 斜率
        'trans_x0': 8.0,  # 时域转换 Sigmoid 中心点
        'c_limit': 0.297,  # 基于 CDF 膝点分析确定
        'balance_p': 1.0  # 均衡因子非线性幂次
    }

    labels_list = df['label'].to_list()
    labels = np.array([
        arr[:height] if len(arr) >= height else np.pad(arr, (0, height - len(arr)), 'constant', constant_values=2)
        for arr in labels_list
    ])

    # --- 2. 因子计算 ---
    # Factor 1: 类别均衡度 (F_bal)
    clutter_ratio = np.mean((labels == 0))
    weather_ratio = np.mean((labels == 1))
    total_coverage = clutter_ratio + weather_ratio
    f_bal_raw = 0.0
    if total_coverage > 0.001:
        # 计算局部相对占比：分母是 total_coverage，而不是 1
        # p_clutter = clutter_ratio / total_coverage
        # p_weather = weather_ratio / total_coverage
        # 基础均衡度：衡量回波内部的成分纯度 (0:极纯, 1:极混)
        base_balance = 1 - abs(clutter_ratio - weather_ratio)
        # base_balance = 1 - abs(p_clutter - p_weather)
        # 论文中的公式，但绘图效果不如激进方案
        # f_bal_raw = (base_balance * total_coverage) ** diff_params['balance_p']
        # 激进优化：引入覆盖度饱和阈值
        # 物理含义：当覆盖度达到 c_limit 时，我们就认为信息量饱和了，不再因为覆盖度小而惩罚它
        saturation_term = np.clip(total_coverage / diff_params['c_limit'], 0, 1)
        f_bal_raw = base_balance * saturation_term

    # Factor 2: 垂直包络重叠 (F_overlap)
    f_overlap_raw = 0.0
    if labels.shape[0] >= diff_params['window_size']:
        overlaps = []
        for i in range(0, labels.shape[0] - diff_params['window_size'] + 1, diff_params['overlap_step']):
            w_labels = labels[i: i + diff_params['window_size'], :]
            c_h = np.where(w_labels == 0)[1]
            w_h = np.where(w_labels == 1)[1]
            if c_h.size > 20 and w_h.size > 20:
                overlaps.append(max(0, np.percentile(c_h, 95) - np.percentile(w_h, 5)))
        if overlaps:
            f_overlap_raw = np.clip(np.max(overlaps) / diff_params['overlap_h_limit'], 0, 1)

    # Factor 3: 时域不稳定性 (F_trans)
    ground_labels = labels[:, 5:diff_params['near_ground_h']]
    dominant_series = []
    for t in range(ground_labels.shape[0]):
        c = np.bincount(ground_labels[t, ground_labels[t] < 2], minlength=2)
        dominant_series.append(0 if c[0] > c[1] else 1 if c[1] > c[0] else -1)

    valid_s = np.array([s for s in dominant_series if s != -1])
    max_trans = 0
    if valid_s.size >= diff_params['window_size']:
        for i in range(len(valid_s) - diff_params['window_size'] + 1):
            max_trans = max(max_trans, np.count_nonzero(np.diff(valid_s[i: i + diff_params['window_size']])))

    f_trans_raw = 1 / (1 + np.exp(-diff_params['trans_k'] * (max_trans - diff_params['trans_x0'])))

    # --- 3. 辅助因子与聚合 ---
    def get_val(feat):
        v = df[feat].to_list()
        return float(v[0][0]) if v and isinstance(v[0], (list, np.ndarray)) else float(v[0]) if v else np.nan

    def _norm(v, f):
        mi, ma = valid_ranges[f]
        return np.clip((v - mi) / (ma - mi), 0, 1) if not pd.isna(v) and ma != mi else 0.0

    f_tem = _norm(get_val('TEM'), 'TEM')
    f_rhu = _norm(get_val('RHU'), 'RHU')
    f_pre = _norm(get_val('PRE'), 'PRE')

    # 计算最终得分
    difficulty = (
            weights.get('balance', 0) * f_bal_raw +
            weights.get('max_overlap', 0) * f_overlap_raw +
            weights.get('temporal_transition', 0) * f_trans_raw +
            weights.get('TEM', 0) * f_tem +
            weights.get('RHU', 0) * f_rhu +
            weights.get('PRE', 0) * f_pre
    )
    # 封装详细诊断信息
    diagnostics = {
        'raw_balance': base_balance if total_coverage > 0.01 else 0.0,  # 新增
        'total_coverage': total_coverage,  # 新增
        'f_balance': f_bal_raw,
        'f_overlap': f_overlap_raw,
        'f_transition': f_trans_raw,
        'f_tem': f_tem,
        'f_rhu': f_rhu,
        'f_pre': f_pre,
        'max_trans_count': max_trans
    }
    return {'difficulty': np.clip(difficulty, 0, 1), 'diagnostics': diagnostics}

#  完全恢复 baseline 的 split_data_files 复杂采样逻辑
def split_data_files(config):
    print("基于采样策略划分数据集...")
    dataset_config = config['dataset']
    root_dir = Path(dataset_config['root_dir'])
    stations = dataset_config.get('stations', [])
    if not stations:
        raise ValueError("配置文件中未指定config['dataset']['stations']")
    # 增加对 difficulty_weights 的健壮性检查
    weights_check = dataset_config.get('difficulty_weights', {})
    if not weights_check or all(v == 0 for v in weights_check.values()):
        print("\n" + "=" * 80)
        print("!!! WARNING: 'difficulty_weights' in config.yaml is missing, empty, or all values are zero.")
        print("!!! This will result in all difficulty scores being 0.")
        print("!!! Please check your config file. Proceeding with all scores as 0...")
        print("=" * 80 + "\n")

    all_train_files, all_val_files = [], []
    train_difficulties, val_difficulties = {}, {}
    all_diagnostics = []  # 新增：用于收集诊断信息
    for station in stations:
        date_range = pd.date_range(start=dataset_config['start'], end=dataset_config['end'], freq='D')
        station_files = sorted([p for p in [
            root_dir / station / f"{d.year}" / f"{d.year}{d.month:02d}" / f"{station}_{d.year}{d.month:02d}{d.day:02d}.parquet"
            for d in date_range] if p.exists()])

        if not station_files:
            print(f"Warning: No Parquet files found for station {station} in the given date range.")
            continue

        month_map = defaultdict(list)
        for file_path in station_files:
            try:
                date = pd.Timestamp(file_path.stem.split('_')[1])
                month_map[date.to_period('M')].append(file_path)
            except (IndexError, ValueError):
                print(f"Warning: Could not parse date from filename: {file_path.name}")
                continue

        np.random.seed(dataset_config['monthly_split']['seed'])
        season_map = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'summer', 6: 'summer', 7: 'summer',
                      8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'winter', 12: 'winter'}

        for month, files_in_month in tqdm(month_map.items(), desc=f"Station {station} months"):
            n_files = len(files_in_month)
            if n_files == 0: continue

            # 修改：接收字典输出，并分离难度分和诊断信息
            difficulty_data = [_compute_file_difficulty(f, dataset_config) for f in files_in_month]
            base_weights = np.array([d['difficulty'] for d in difficulty_data])
            # 新增：收集诊断信息
            for i, f in enumerate(files_in_month):
                diag_data = difficulty_data[i]['diagnostics']
                diag_data['file_path'] = str(f)
                all_diagnostics.append(diag_data)

            season = season_map[month.month]
            season_factors = dataset_config.get('season_factors',
                                                {'summer': 1.5, 'winter': 0.8, 'spring': 1.0, 'autumn': 1.2})
            factor = season_factors.get(season, 1.0)
            factor_alpha = 4.0 #样本难度得分放大指数
            adjusted_weights = (base_weights ** factor_alpha) * factor
            total_weight = np.sum(adjusted_weights)
            probabilities = adjusted_weights / total_weight if total_weight > 0 else np.full(n_files, 1 / n_files)
            all_indices = np.arange(n_files)
            prioritized_indices = np.random.choice(all_indices, size=n_files, replace=False, p=probabilities)
            train_count = int(n_files * dataset_config['train_ratio'])
            train_indices = prioritized_indices[:train_count]
            val_indices = prioritized_indices[train_count:]

            current_month_train_files = [files_in_month[i] for i in train_indices]
            current_month_val_files = [files_in_month[i] for i in val_indices]

            all_train_files.extend(current_month_train_files)
            all_val_files.extend(current_month_val_files)

            for i in train_indices:
                train_difficulties[str(files_in_month[i])] = difficulty_data[i]
            for i in val_indices:
                val_difficulties[str(files_in_month[i])] = difficulty_data[i]
    # 新增：将诊断信息保存到CSV文件
    if all_diagnostics:
        diag_df = pd.DataFrame(all_diagnostics)
        diag_output_path = Path("./采样策略样本划分诊断信息.csv")
        diag_df.to_csv(diag_output_path, index=False)
        print(f"诊断信息已保存到 {diag_output_path}")
    return all_train_files, all_val_files, train_difficulties, val_difficulties

def _save_validation_dates(val_files, output_path):
    if not val_files:
        print("Warning: 验证集列表为空，没有需要保存的日期.")
        return
    validation_entries = set()
    for file_path in val_files:
        try:
            parts = Path(file_path).stem.split('_')
            station_id = parts[0]
            date_str = parts[1]
            pd.to_datetime(date_str, format='%Y%m%d')
            validation_entries.add((station_id, date_str))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse station/date from filename: {Path(file_path).name}")
            continue
    if not validation_entries:
        print("Warning: No valid entries could be extracted from validation files.")
        return
    sorted_entries = sorted(list(validation_entries))
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for station, date in sorted_entries:
                f.write(f"{station},{date}\n")
        print(f"Validation station-date pairs successfully saved to {output_path}")
    except IOError as e:
        print(f"Error: Could not write validation dates to {output_path}. Reason: {e}")

H_LOW_INDEX = 150       # 假设 3km 对应高度索引 150 (请根据您的雷达分辨率调整)
Z_ECHO_THRESHOLD = -40.0 # 定义有效回波的 Z 值阈值 (dBZ)
# [GEMINI] 为新特征定义近似的归一化参数 (这些值可以根据经验调整)
# 假设 Z 值的标准差通常不会超过 20 dBZ
LOW_Z_STD_MAX = 20.0
class SeasonalRadarDataset(Dataset):
    """
    自定义的数据集类，负责加载、预处理和提供模型所需的样本。
    """
    def __init__(self, cfg, file_list, mode='train', radar_normalizer=None, ground_normalizer=None):
        """
        构造函数。
        Args:
            cfg (dict): 'dataset'部分的配置字典。
            file_list (list): 该数据集要处理的文件路径列表。
            mode (str): 'train' 或 'val'。
            radar_normalizer (Normalizer): 用于雷达数据的归一化器实例。
            ground_normalizer (Normalizer): 用于地面数据的归一化器实例。
        """
        self.config = cfg['dataset']
        self.file_list = sorted(file_list)
        self.mode = mode
        self.model_channels = cfg['model']['in_ch']
        self.model_name = cfg['model']['name'].lower()
        self.radar_normalizer = radar_normalizer
        self.ground_normalizer = ground_normalizer
        self.window_size = self.config['window_size']
        self.stride = self.config['stride'] if mode == 'train' else self.config['window_size']
        self.height = self.config['height']
        self.h_true_max = self.config.get('h_true_max_bin', 200.0)
        self.resample_rate = self.config.get('resample_rate_minutes', 1)  # 从配置读取降采样率
        self._precompute_metadata()
        self.valid_windows = self._generate_windows()
        self.gate_size_m = 30.0

    def _precompute_metadata(self):
        """预计算每个文件的长度，用于快速索引。"""
        self.file_lengths = [pq.read_metadata(f).num_rows for f in self.file_list]
        self.cumulative_lengths = np.cumsum([0] + self.file_lengths)

    def _generate_windows(self):
        """根据滑动窗口大小和步长，生成所有有效的样本窗口。"""
        windows = []
        for i, n_samples in enumerate(self.file_lengths):
            if n_samples >= self.window_size:
                for s in range(0, n_samples - self.window_size + 1, self.stride):
                    windows.append((self.cumulative_lengths[i] + s, i))
        return windows

    def __len__(self):
        """返回数据集中有效样本的数量。"""
        return len(self.valid_windows)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个样本。这是数据加载的核心。
        """
        # 1. 定位数据窗口
        global_start_idx, file_idx = self.valid_windows[idx]
        local_start_idx = global_start_idx - self.cumulative_lengths[file_idx]
        file_path = str(self.file_list[file_idx])
        # 使用文件缓存管理器
        partition_df = _read_parquet_cached(file_path)
        window = partition_df.iloc[local_start_idx: local_start_idx + self.window_size]

        # 2. 处理雷达数据流
        radar_features_list = ['Z', 'V', 'W', 'LDR']
        radar_features_raw = np.stack([self._process_feature(window[feat]) for feat in radar_features_list], axis=-1)

        if self.radar_normalizer:
            radar_features_norm = self.radar_normalizer.transform(radar_features_raw)  # (T, H, 4)
        else:
            radar_features_norm = radar_features_raw  # 如果没有归一化器

        # 2a. 创建归一化的“高度”特征 (H_norm)
        if self.model_channels > 4: #是否启用了除4个雷达特征外的其他特征
            h_norm_feature = np.arange(self.height, dtype=np.float32) / self.height
            h_norm_broadcast = np.broadcast_to(h_norm_feature.reshape(1, self.height, 1),
                                               (self.window_size, self.height, 1))
            # 5. 最终堆叠: (T, H, 4) + (T, H, 1) = (T, H, 5)
            radar_features = np.concatenate([radar_features_norm, h_norm_broadcast], axis=-1)
        else:
            radar_features = radar_features_norm

        # 3. 处理地面数据流
        if 'surf' in self.model_name:
            ground_features_list = ['TEM', 'RHU', 'PRE']
            ground_features_ts = window[ground_features_list].values.astype(np.float32)
            # 3a. 对地面数据进行降采样
            if self.resample_rate > 1:
                num_timesteps, num_features = ground_features_ts.shape
                new_num_timesteps = num_timesteps // self.resample_rate
                if new_num_timesteps > 0:
                    truncated_ts = ground_features_ts[:new_num_timesteps * self.resample_rate]
                    ground_features_ts = truncated_ts.reshape(new_num_timesteps, self.resample_rate, num_features).mean(
                        axis=1)

            # 3b. 归一化地面数据
            if self.ground_normalizer:
                ground_features_ts = self.ground_normalizer.transform(ground_features_ts)
            physics_features_ts = ground_features_ts.astype(np.float32)
            physics_features = torch.from_numpy(physics_features_ts).float()
        else:
            physics_features = torch.zeros(1)
        # 4. 处理标签和LCL真值
        labels = self._process_labels(window['label'])
        # A. 修改回归真值：使用 H_true_conn
        # 注意：这里读取的是预处理脚本中新生成的 'H_true_conn' 列
        # 如果Parquet中存在NaN，保持NaN，后续 Loss 计算时会自动 mask 掉
        h_true_raw = window['H_true_conn'].values
        h_true_sequence = h_true_raw / self.h_true_max  # 归一化到 [0, 1]
        # B. LCL 锚点 (ech_utils返回的是 米，必须除以30转为Bin)
        lcl_meters = window['LCL'].values
        # 换算: 米 -> Bin
        lcl_bins = lcl_meters / self.gate_size_m
        # 处理 LCL 的 NaN (虽然预处理通常会填补，但为了安全填0或保持)
        # lcl_sequence = lcl_bins / self.h_true_max  # 除以 500，导致值太小
        # 新代码建议：固定分母，放大信号
        # 设定 LCL 参考最大值为 200 个库 (约 6000米)，这足够覆盖绝大多数 LCL 情况
        LCL_REF_MAX_BIN = 180.0
        lcl_sequence = lcl_bins / LCL_REF_MAX_BIN
        # 极值处理：防止 LCL 超过最大高度导致值 > 1
        lcl_sequence = np.clip(lcl_sequence, 0.0, 1.0)
        # NaN 处理：保持 0
        lcl_sequence = np.nan_to_num(lcl_sequence, nan=0.0)
        return {
            'radar_features': torch.from_numpy(radar_features).float(),
            'physics_features': physics_features,
            'labels': torch.from_numpy(labels).long(),
            'h_true': torch.from_numpy(h_true_sequence).float(),
            'lcl': torch.from_numpy(lcl_sequence).float()
        }

    def _process_feature(self, feature_series):
        """辅助函数：处理单个雷达特征列，确保高度一致。"""
        processed = [
            arr[:self.height] if len(arr) >= self.height else np.pad(arr, (0, self.height - len(arr)), 'constant',
                                                                     constant_values=-100.0) for arr in feature_series]
        return np.vstack(processed)

    def _process_labels(self, label_series):
        """辅助函数：处理标签列，确保高度一致。"""
        processed = [
            arr[:self.height] if len(arr) >= self.height else np.pad(arr, (0, self.height - len(arr)), 'constant',
                                                                     constant_values=2) for arr in label_series]
        return np.vstack(processed)

def create_loaders(config_path,val_dates_output_path=None, force_refit=False,force_resplit=False):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
        # 从配置中获取关键参数，特别是 height
    dataset_config = config['dataset']
    height_cutoff = dataset_config['height']
    split_cache_path = Path("./datasplit.npz")
    #  恢复 datasplit_difficulties.npz 的路径定义
    difficulty_cache_path = Path("./datasplit_difficulties.npz")
    if not force_resplit and split_cache_path.exists():
        print(f"Loading cached data split from {split_cache_path}...")
        with np.load(split_cache_path, allow_pickle=True) as data:
            train_files, val_files = [Path(p) for p in data['train_files']], [Path(p) for p in data['val_files']]
    else:
        print("No split cache found or forcing resplit, performing data split...")
        #  接收并处理难度分数
        train_files, val_files, train_diff, val_diff = split_data_files(config)

        np.savez(split_cache_path, train_files=[str(p) for p in train_files], val_files=[str(p) for p in val_files])
        print(f"Data split saved to {split_cache_path}")

        #  恢复保存 datasplit_difficulties.npz
        np.savez(difficulty_cache_path, train_difficulties=train_diff, val_difficulties=val_diff)
        print(f"Difficulty scores saved to {difficulty_cache_path}")

    print(f"Data split loaded: {len(train_files)} training files, {len(val_files)} validation files.")
    if val_dates_output_path:
        _save_validation_dates(val_files, output_path=val_dates_output_path)

    # --- 双归一化器逻辑 ---
    radar_features_list = ['Z', 'V', 'W', 'LDR']
    ground_features_list = ['TEM', 'RHU', 'PRE']
    radar_normalizer_path = Path("./radar_normalizer.pkl")
    ground_normalizer_path = Path("./ground_normalizer.pkl")

    radar_normalizer = Normalizer(feature_order=radar_features_list)
    ground_normalizer = Normalizer(feature_order=ground_features_list)

    if force_refit or not radar_normalizer_path.exists() or not ground_normalizer_path.exists():
        print("Fitting new normalizers using memory-efficient partial_fit. This may take a while...")
        if not train_files: raise ValueError("Training file list is empty, cannot fit normalizer.")

        # 为 partial_fit 重新初始化 Scaler
        radar_normalizer.pipeline.named_steps['scaler'].__init__()
        ground_normalizer.pipeline.named_steps['scaler'].__init__()

        # 分块处理文件进行拟合(防止内存不够）
        for file_path in tqdm(train_files, desc="Fitting normalizers chunk-by-chunk"):
            df = pd.read_parquet(file_path)
            # --- 处理雷达数据 ---
            # 1. 定义与 _process_feature 相同逻辑的辅助函数
            def standardize_height(feature_series, height):
                processed = [
                    arr[:height] if len(arr) >= height else np.pad(arr, (0, height - len(arr)), 'constant',
                                                                   constant_values=-100.0)
                    for arr in feature_series.values
                ]
                return np.vstack(processed) if processed else np.array([]).reshape(0, height)

            # 2. 对每个雷达特征应用高度标准化
            # 在stack前检查列表是否为空
            # if not all(chunk.size > 0 for chunk in radar_chunk_list): continue
            radar_chunk_list = [standardize_height(df[feat], height_cutoff) for feat in radar_features_list]
            radar_chunk = np.stack(radar_chunk_list, axis=-1)
            radar_chunk_2d = radar_chunk.reshape(-1, len(radar_features_list))

            # 3. 将无效值转为 NaN 并进行增量拟合
            radar_chunk_nan = radar_normalizer.pipeline.named_steps['invalid_to_nan'].transform(radar_chunk_2d)
            valid_radar_data = radar_chunk_nan[~np.isnan(radar_chunk_nan).any(axis=1)]
            if valid_radar_data.shape[0] > 0:
                radar_normalizer.pipeline.named_steps['scaler'].partial_fit(valid_radar_data)

            # --- 处理地面数据 ---
            ground_chunk = df[ground_features_list].values.astype(np.float32)
            valid_ground_data = ground_chunk[~np.isnan(ground_chunk).any(axis=1)]
            if valid_ground_data.shape[0] > 0:
                ground_normalizer.pipeline.named_steps['scaler'].partial_fit(valid_ground_data)
        #手动拟合 SimpleImputer
        dummy_radar_data = np.zeros((1, len(radar_features_list)))
        radar_normalizer.pipeline.named_steps['imputer'].fit(dummy_radar_data)
        dummy_ground_data = np.zeros((1, len(ground_features_list)))
        ground_normalizer.pipeline.named_steps['imputer'].fit(dummy_ground_data)
        # 拟合完成后保存
        radar_normalizer.save(radar_normalizer_path)
        ground_normalizer.save(ground_normalizer_path)
#       打印最终的统计特征
        print("\n--- Normalization Statistics ---")
        def print_stats(normalizer, name):
            scaler = normalizer.pipeline.named_steps['scaler']
            if hasattr(scaler, 'mean_'):
                print(f"[{name}]")
                for i, feature in enumerate(normalizer.feature_order):
                    mean = scaler.mean_[i];
                    std = np.sqrt(scaler.var_[i])
                    print(f"  - {feature}: Mean = {mean:.4f}, Std = {std:.4f}")
            else:
                print(f"[{name}] - Scaler not fitted.")

        print_stats(radar_normalizer, "Radar Features")
        print_stats(ground_normalizer, "Ground Features")
        print("--------------------------------\n")

    else:
        print(f"Loading existing normalizers...")
        radar_normalizer.load(radar_normalizer_path)
        ground_normalizer.load(ground_normalizer_path)

    # --- 创建最终的数据集和加载器 ---
    train_set = SeasonalRadarDataset(config, file_list=train_files, mode='train',
                                     radar_normalizer=radar_normalizer, ground_normalizer=ground_normalizer)
    val_set = SeasonalRadarDataset(config, file_list=val_files, mode='val',
                                   radar_normalizer=radar_normalizer, ground_normalizer=ground_normalizer)
    train_shuffle = True
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=train_shuffle,
                              num_workers=config['train']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], shuffle=False,
                            num_workers=config['train']['num_workers'], pin_memory=True)
    if train_shuffle:
        print(f"本次训练已开启Shuffle")
    normalizers = {'radar': radar_normalizer, 'ground': ground_normalizer}
    return train_loader, val_loader, normalizers

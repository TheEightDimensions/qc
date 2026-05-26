# -*- coding:utf-8 -*-
# author: Wang jia peng
# description: 终极稳定版预处理脚本
# 1. 采用 Inner Join 确保 Parquet 数据结构的纯净性，杜绝底层读取崩溃。
# 2. 整合 LCL 抬升凝结高度标量计算。
# 3. 整合 三种 ECH 策略的批量矩阵计算及平滑处理。
# 4. 支持 geo_encoder 地理特征编码。
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
import yaml
import struct, os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import ast

# --- 导入算法模块 ---
try:
	import ech_utils
except ImportError:
	logging.warning("未找到 ech_utils.py，将跳过新策略 ECH 与 LCL 计算。")

# --- 日志和常量配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
VALID_SURF_RANGES = {'TEM': (-60, 60), 'RHU': (0, 100), 'PRE': (0, 100)}
VALID_LABEL_VALUES = {0, 1, 2}


def safe_convert_label_array(label_str):
	"""高效、安全地将标签数组字符串转换为numpy整数数组。"""
	try:
		if pd.isna(label_str): return None
		if isinstance(label_str, str):
			cleaned_str = label_str.replace(' ', '').strip('[]')
			if not cleaned_str: return None
			arr = np.array([int(x) for x in cleaned_str.split(',')], dtype=np.int8)
			if not set(np.unique(arr)).issubset(VALID_LABEL_VALUES):
				# logging.warning(f"发现无效标签值 (只允许0,1,2)，该行将被丢弃: {arr}")
				return None
			return arr
		return None
	except Exception as e:
		logging.warning(f"标签数据转换警告: '{label_str}' -> {e}")
		return None


def run_process_month_for_task(task_args):
	"""子进程的工作函数，负责处理单个站点的一个月份。"""
	station, year, month, config, station_features_df = task_args
	try:
		processor = RadarProcessor(config=config, station_features_df=station_features_df)
		processor._process_month(station, year, month)
	except Exception as e:
		logging.error(f"处理任务 ({station}, {year}-{month}) 时发生严重错误: {e}", exc_info=True)


class RadarProcessor:
	def __init__(self, config_path=None, config=None, station_features_df=None):
		if config:
			self.config = config
		elif config_path:
			with open(os.path.abspath(config_path), encoding='utf-8') as f:
				self.config = yaml.safe_load(f)
		else:
			raise ValueError("必须提供 config_path 或 config")

		if station_features_df is not None:
			self.station_features_df = station_features_df
		else:
			if config_path:
				self.station_features_df = self._load_station_features()

	def _load_station_features(self):
		try:
			if 'paths' not in self.config or 'station_features' not in self.config['paths']:
				return None
			features_path = self.config['paths']['station_features']
			monthly_features = []
			if 'geo_encoder' in self.config and 'monthly_features' in self.config['geo_encoder']:
				monthly_features = self.config['geo_encoder']['monthly_features']
			converters = {col: ast.literal_eval for col in monthly_features}
			df = pd.read_csv(features_path, dtype={'station': str}, converters=converters)
			df.set_index('station', inplace=True)
			return df
		except Exception:
			return None

	def _load_labels(self, label_path):
		if not label_path.exists():
			return pd.DataFrame()
		df = pd.read_csv(
			label_path, parse_dates=['D_DATETIME'],
			converters={'label': safe_convert_label_array}
		)
		df.dropna(subset=['label'], inplace=True)
		return df.rename(columns={'D_DATETIME': 'timestamp'})

	def _load_surf(self, surf_path):
		if not surf_path.exists():
			return pd.DataFrame()
		try:
			df = pd.read_csv(surf_path, usecols=['Datetime', 'TEM', 'RHU', 'PRE'],
							 parse_dates=['Datetime'], encoding='gbk')
			df.drop_duplicates(subset=['Datetime'], keep='first', inplace=True)
			for col, (min_val, max_val) in VALID_SURF_RANGES.items():
				df.loc[~df[col].between(min_val, max_val, inclusive='both'), col] = np.nan
			df['is_original'] = np.where(df[['TEM', 'RHU', 'PRE']].notna().all(axis=1), 1, 0)
			df_resampled = df.set_index('Datetime').resample('1min').asfreq()
			filled_data = df_resampled[['TEM', 'RHU', 'PRE']].ffill().bfill()
			filled_data['is_original'] = df_resampled['is_original'].fillna(0)
			filled_data.dropna(how='all', inplace=True)
			return filled_data.reset_index()
		except Exception as e:
			logging.error(f"地面数据加载或处理失败: {surf_path} -> {e}")
			return pd.DataFrame()

	def _find_bin_file(self, bin_path, station, time_key):
		bin_files = list(bin_path.glob(f"Z_RADA_I_{station}_{time_key}_O_YCCR_*_RAW_MM.BIN"))
		if bin_files:
			return bin_files[0]
		raise FileNotFoundError(f"未找到匹配的BIN文件: station={station}, time={time_key}")

	def _process_day(self, station, date, monthly_labels_df):
		if monthly_labels_df.empty: return
		labels_today = monthly_labels_df[monthly_labels_df['timestamp'].dt.date == date.date()].copy()
		if labels_today.empty: return

		day_str = date.strftime("%Y%m%d")
		next_day = date + pd.Timedelta(days=1)
		surf_path = Path(self.config['paths']['raw_surf'].format(station=station, start_time=f"{day_str}000000",
																 end_time=f"{next_day.strftime('%Y%m%d')}000000"))
		surf_df = self._load_surf(surf_path)
		if surf_df.empty: return

		# 核心基座：采用 Inner Join 保证后续 label 数据 100% 都是纯正数组
		merged_df = pd.merge(labels_today, surf_df, left_on='timestamp', right_on='Datetime', how='inner')
		if merged_df.empty: return

		records = []
		bin_path = Path(self.config['paths']['raw_radar'].format(station=station, day=day_str))

		# 标记修改开始：预扫描建立当日 BIN 文件的哈希索引，O(N)极速寻址
		bin_file_index = {}
		if bin_path.exists():
			for f_path in bin_path.glob(f"Z_RADA_I_{station}_*_O_YCCR_*_RAW_MM.BIN"):
				try:
					t_key = f_path.name.split('_')[4]
					bin_file_index[t_key] = f_path
				except IndexError:
					continue
		# 标记修改结束

		# 准备 geo_feature_vector
		geo_feature_vector = None
		if self.station_features_df is not None and 'geo_encoder' in self.config:
			try:
				station_series = self.station_features_df.loc[station]
				month_idx = date.month - 1
				geo_feature_vector_list = [
					station_series[name][month_idx] if name in self.config['geo_encoder']['monthly_features'] else
					station_series[name] for name in self.config['geo_encoder']['input_features']]
				geo_feature_vector = np.array(geo_feature_vector_list, dtype=np.float32)
			except (KeyError, IndexError):
				pass

		# --- 遍历分钟级数据 ---
		for _, row in merged_df.iterrows():
			try:
				time_key = row['timestamp'].strftime("%Y%m%d%H%M%S")

				# 标记修改开始：使用哈希索引代替原本极度耗时的 _find_bin_file
				bin_file = bin_file_index.get(time_key)
				if not bin_file:
					raise FileNotFoundError(f"未找到匹配的BIN文件: station={station}, time={time_key}")
				# 标记修改结束

				radar_data = self.read_bin(bin_file)

				if any(field not in radar_data for field in ['Z1', 'V1', 'W1']): continue
				if 'LDR' not in radar_data:
					radar_data['LDR'] = np.full_like(radar_data['Z1'], -100.0, dtype=np.float32)

				# 标记修改开始：强制对齐 label 维度与 Z1 数组的长度完全一致
				label_arr = row['label']
				if not isinstance(label_arr, np.ndarray): continue
				z_len = len(radar_data['Z1'])
				if len(label_arr) > z_len:
					label_arr = label_arr[:z_len]
				elif len(label_arr) < z_len:
					label_arr = np.pad(label_arr, (0, z_len - len(label_arr)), 'constant', constant_values=-1)
				# 标记修改结束

				# 标记新增：单点计算 LCL (每分钟计算一次)
				if 'ech_utils' in globals():
					try:
						lcl_height = ech_utils.calculate_lcl_bolton(row['TEM'], row['RHU'])
						lcl_height = np.round(lcl_height, 2) if not np.isnan(lcl_height) else np.nan
					except Exception:
						lcl_height = np.nan
				else:
					lcl_height = np.nan

				record_data = {
					'station': station, 'timestamp': row['timestamp'], 'Z': radar_data['Z1'], 'V': radar_data['V1'],
					'W': radar_data['W1'], 'LDR': radar_data['LDR'],
					'label': label_arr,  # 标记修改开始：替换为维度修剪安全的 label_arr
					'TEM': row['TEM'], 'RHU': row['RHU'], 'PRE': row['PRE'],
					'LCL': lcl_height,
					'is_original': np.uint8(row['is_original'])
				}
				if geo_feature_vector is not None:
					record_data['geo_features'] = geo_feature_vector
				records.append(record_data)

			except FileNotFoundError:
				continue
			except Exception as e:
				logging.error(f"处理文件 {time_key} for {station} 时出错: {e}")

		# 标记修改开始：1. 修正了缩进Bug(此块必须平级位于for循环外)；2. Stack前增加 max_h 对齐双保险
		if records and 'ech_utils' in globals():
			try:
				# --- 强制对齐双保险：防止设备中途调整库数导致 stack 崩溃 ---
				max_h = max(len(r['label']) for r in records)
				aligned_labels = []
				for r in records:
					arr = r['label']
					if len(arr) < max_h:
						aligned_labels.append(np.pad(arr, (0, max_h - len(arr)), 'constant', constant_values=-1))
					else:
						aligned_labels.append(arr[:max_h])

				# np.stack 此时绝对安全
				label_matrix = np.stack(aligned_labels)
				timestamp_series = pd.Series([r['timestamp'] for r in records])

				GATE_SIZE = 30
				# 矩阵化调用连通域算法
				ech_conn_raw = ech_utils.calculate_ech_connectivity(label_matrix, gate_size_m=GATE_SIZE)

				# 时序平滑
				smooth_min = '5min'
				ech_conn = ech_utils.smooth_ech_series(timestamp_series, ech_conn_raw, window_minutes=smooth_min)

				# 回填至记录表
				for i, r in enumerate(records):
					r['H_true_conn'] = np.round(ech_conn[i], 2) if not np.isnan(ech_conn[i]) else np.nan

			except Exception as e:
				logging.error(f"ECH连通域策略批量计算失败: {e}")
				for r in records:
					r['H_true_conn'] = np.nan
		elif records:
			for r in records:
				r['H_true_conn'] = np.nan
		# 标记修改结束

		# --- 写入 Parquet ---
		if records:
			df_to_save = pd.DataFrame(records)
			year_str, month_str = date.strftime("%Y"), date.strftime("%Y%m")
			output_dir = Path(self.config['dataset']['root_dir']) / station / year_str / month_str
			output_dir.mkdir(parents=True, exist_ok=True)
			pq.write_table(pa.Table.from_pandas(df_to_save), output_dir / f"{station}_{day_str}.parquet",
						   compression='snappy')

	def _process_month(self, station, year, month):
		month_str = f"{year}{month:02d}"
		label_path = Path(self.config['paths']['std_labels'].format(station=station, yearmonth=month_str))
		monthly_labels_df = self._load_labels(label_path)
		if monthly_labels_df.empty:
			logging.warning(f"站点 {station} 在 {month_str} 的标签数据为空，跳过整个月。")
			return

		days_in_month = pd.date_range(start=f'{year}-{month}-01',
									  end=pd.to_datetime(f'{year}-{month}-01') + pd.offsets.MonthEnd(0), freq='D')
		for day in days_in_month:
			self._process_day(station, day, monthly_labels_df)

	def read_bin(self, file_path):
		bin_dict = {}
		try:
			with open(file_path, 'rb') as f:
				if os.path.getsize(file_path) < 500: return {}
				f.seek(40)
				sta_name = struct.unpack("24s", f.read(24))[0].decode('utf-8', errors='ignore').strip('\x00')
				bin_dict['lat'] = struct.unpack('f', f.read(4))[0]
				bin_dict['lon'] = struct.unpack('f', f.read(4))[0]
				bin_dict['ata_hgt'] = struct.unpack('f', f.read(4))[0]
				bin_dict['altitude'] = struct.unpack('f', f.read(4))[0]
				bin_dict['sta_name'] = sta_name
				f.seek(76, 1)
				bin_dict['dis_slt'] = struct.unpack('H', f.read(2))[0]
				f.seek(238, 1)
				cut_num = struct.unpack('i', f.read(4))[0]
				bin_dict['cut_num'] = cut_num
				if cut_num > 1: logging.warning(f"扫描层数不为1: {os.path.basename(file_path)}")
				f.seek(112, 1)
				f.seek(68, 1)
				bin_dict['vmax'] = struct.unpack('f', f.read(4))[0]
				f.seek(376 - 112 - 68 - 4, 1)
				bin_dict['type_num'] = struct.unpack('H', f.read(2))[0]
				f.seek(54, 1)
				return self.get_radial_data(bin_dict, f)
		except struct.error:
			return {}
		except Exception:
			return {}

	def get_radial_data(self, bin_dict, f):
		type_code = []
		type_number = bin_dict['type_num']
		with f:
			for _ in range(type_number):
				typeHeader = {"Data_Type": struct.unpack('H', f.read(2)), "Scale": struct.unpack('H', f.read(2)),
							  "Offset": struct.unpack('H', f.read(2)), "Bin Bytes": struct.unpack('H', f.read(2)),
							  "Bin number": struct.unpack('H', f.read(2)), "Flags": struct.unpack('h', f.read(2)),
							  "Length": struct.unpack('i', f.read(4)), "Reserved": f.read(16)}
				bin_dict['bin_num'] = typeHeader["Bin number"][0]
				Data_Type = typeHeader["Data_Type"][0]
				type_code.append(Data_Type)
				block_length = typeHeader["Length"][0]
				offset = typeHeader["Offset"][0]
				scale = typeHeader["Scale"][0]
				bin_length = typeHeader["Bin Bytes"][0]
				data_body = f.read(block_length)
				raw = np.frombuffer(data_body, 'u' + str(bin_length)).astype(np.float64)
				if Data_Type == 43:
					value = (raw - offset) / scale
				else:
					value = np.full(raw.shape, -100.0, dtype=float)
					mask = (raw == 0) | (raw == 1)
					value[~mask] = (raw[~mask] - offset) / scale
				if Data_Type in self.config['data_type']:
					data_name = self.config['data_type'][Data_Type]
					bin_dict[data_name] = value
		bin_dict['type_code'] = type_code
		return bin_dict

	def _get_last_processed_date(self, station):
		root_dir = Path(self.config['dataset']['root_dir'])
		station_dir = root_dir / str(station)
		if not station_dir.exists():
			return pd.to_datetime(self.config['start']) - pd.Timedelta(days=1)
		parquet_files = list(station_dir.glob('**/*.parquet'))
		if not parquet_files:
			return pd.to_datetime(self.config['start']) - pd.Timedelta(days=1)
		dates = [pd.to_datetime(f.stem.split('_')[1], format='%Y%m%d') for f in parquet_files if f.stem.split('_')]
		if not dates:
			return pd.to_datetime(self.config['start']) - pd.Timedelta(days=1)
		return max(dates)

	def process_all_parallel(self, incremental_update=False):
		logging.info("启动并行处理...")
		main_processor = RadarProcessor(config_path=os.path.abspath("../configs/config.yaml"))
		config = main_processor.config
		station_features_df = main_processor.station_features_df
		if station_features_df is None: return

		tasks = []
		config_end_date = pd.to_datetime(config['end'])

		for station in config['stations']:
			start_date = self._get_last_processed_date(station) + pd.Timedelta(
				days=1) if incremental_update else pd.to_datetime(config['start'])
			if start_date > config_end_date:
				logging.info(f"站点 {station} 的数据已是最新，无需更新。")
				continue
			logging.info(f"站点 {station} 将从 {start_date.strftime('%Y-%m-%d')} 开始处理。")
			date_range_for_station = pd.date_range(start_date, config_end_date)
			if date_range_for_station.empty: continue

			unique_months = sorted(list(set([(d.year, d.month) for d in date_range_for_station])))
			for year, month in unique_months:
				tasks.append((str(station), year, month, config, station_features_df))

		if not tasks:
			logging.info("没有需要处理的新任务。")
			return

		num_processes = min(6, cpu_count())
		logging.info(f"总任务数: {len(tasks)} (站-月), 使用 {num_processes} 个进程。")
		with Pool(processes=num_processes) as pool:
			list(tqdm(pool.imap_unordered(run_process_month_for_task, tasks), total=len(tasks), desc="总进度"))
		logging.info("所有处理任务已完成。")


if __name__ == "__main__":
	INCREMENTAL_UPDATE = True
	processor = RadarProcessor(config_path="../configs/config.yaml")
	processor.process_all_parallel(incremental_update=INCREMENTAL_UPDATE)
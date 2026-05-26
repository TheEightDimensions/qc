# -*- coding:utf-8 -*-
# author: Wang jia peng
# create: 2025/8/11 18:14
# filename: visualize_inference.py
# 生成验证集的推理图像，并统计评估指标
# 标签使用std_labels
import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import rcParams
from scipy.ndimage import binary_opening
import multiprocessing
import os
from collections import defaultdict
import glob
# --- 导入新架构的模型 ---
from model import BaselineModel, BaselineSurfModel, BaselineSurfLCLModel
from normalize import Normalizer
from preprocess import RadarProcessor
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- 1. 后处理函数---
def post_process_with_dynamic_lcl(z_data, lcl_predictions_bins, safety_margin_bins, height_cutoff,
								  structure_size=(6, 6)):
	"""
	使用动态LCL预测作为阈值，并以更高效率的方式进行后处理。
	该版本采纳了先对数据进行高度截断，再执行计算密集型操作的优化建议。
	Args:
		z_data (np.ndarray): 经过模型初步QC后的二维（时间, 高度）反射率数据。
		lcl_predictions_bins (np.ndarray): 模型预测的LCL高度一维序列，单位为距离库索引。
		safety_margin_bins (int): 在LCL预测高度上增加的安全裕度，单位为距离库。
		height_cutoff (int): 模型的最大处理高度（例如500），作为硬性上限。
		structure_size (tuple): 形态学开运算使用的滤波器大小。
	Returns:
		np.ndarray: 经过智能后处理后的反射率数据。
	"""
	# 创建一个最终数据的副本，用于保存最终结果
	final_z_data = z_data.copy()
	# --- 步骤 1: 预先截断数据以提高效率 ---
	# 只截取模型处理高度以下的数据块进行操作。这是一个视图（view），操作会影响 final_z_data
	low_level_block = final_z_data[:, :height_cutoff]
	# --- 步骤 2: 在截断后的小数据块上，识别所有潜在的噪声点 ---
	# 将截断后的数据块转换为二进制掩码（有回波为True，NaN为False）
	binary_mask_2d = ~np.isnan(low_level_block)
	# 定义形态学开运算的结构元素（滤波器）
	structure = np.ones(structure_size, dtype=int)
	# 在小数据块上执行开运算，这一步的效率被大大提高
	cleaned_mask_2d = binary_opening(binary_mask_2d, structure=structure)
	# 找到被开运算移除的像素点，这些是“潜在的噪声”
	potential_noise_mask = binary_mask_2d & ~cleaned_mask_2d
	# --- 步骤 3: 同样在小数据块上，构建动态的“允许处理”区域 ---
	# 获取时间步长
	total_time = low_level_block.shape[0]
	# 创建一个与小数据块同样大小的、全为False的掩码
	processing_zone_mask = np.zeros_like(low_level_block, dtype=bool)
	# 逐个时间步构建动态处理高度上限
	for t in range(total_time):
		# 获取当前时间点的LCL预测值
		lcl_t = lcl_predictions_bins[t]
		# 检查LCL预测值是否有效
		if not np.isnan(lcl_t):
			# 计算包含安全裕度的处理高度上限
			threshold_t = int(lcl_t) + safety_margin_bins
			# 确保上限不超过截断高度 height_cutoff
			threshold_t = min(threshold_t, height_cutoff)
			# 在小掩码的对应列，将上限以下的区域标记为“允许处理”
			processing_zone_mask[t, :threshold_t] = True
	# --- 步骤 4: 应用约束，找到最终要剔除的噪声点 ---
	# 取交集：最终要剔除的点 = 潜在的噪声 AND 位于允许处理区域内
	final_noise_to_remove_mask = potential_noise_mask & processing_zone_mask
	# --- 步骤 5: 更新数据 ---
	# 在截断的数据块视图中，将最终确定的噪声点设置为NaN
	# 由于 low_level_block 是 final_z_data 的一个视图，这里的修改会自动反映到 final_z_data 中
	low_level_block[final_noise_to_remove_mask] = np.nan
	# 返回修改后的、与原始数据同样高度的最终结果
	return final_z_data
# --- 2. 绘图函数 (无修改) ---
def plot_z_panel(fig, ax, data_values, timestamps, height_levels, title,ech_values=None):
	rcParams['font.sans-serif'] = ['SimHei']
	rcParams['axes.unicode_minus'] = False
	# beijing_timestamps = [ts + timedelta(hours=8) for ts in timestamps]
	colors = ['#042ac9', '#0852d1', '#0c7ad5', '#01a0f6', '#00ecec', '#00d800', '#019000',
			  '#ffff00', '#e7c000', '#ff9000', '#ff0000', '#d60000']
	bounds = [-30, -20, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
	cmap = ListedColormap(colors)
	cmap.set_under('#000080')  # 小于 -30 的颜色
	cmap.set_bad('white', alpha=0)
	norm = BoundaryNorm(bounds, cmap.N)
	masked_values = np.ma.masked_invalid(data_values.T)
	c = ax.pcolormesh(timestamps, height_levels, masked_values, cmap=cmap, norm=norm, shading='auto')
	# 绘制ECH曲线
	if ech_values is not None:
		# 过滤掉 NaN 值以避免绘图中断
		valid_indices = ~np.isnan(ech_values)
		if np.any(valid_indices):
			# 将 timestamps 和 ech_values 转换为 numpy 数组以便索引
			ts_array = np.array(timestamps)
			# beijing_ts_array = ts_array + np.timedelta64(8, 'h')
			ech_array = np.array(ech_values)
			ax.plot(ts_array[valid_indices], ech_array[valid_indices],
					color='red', linestyle='-', linewidth=1, label='Predicted ECH')
		# 可选：添加图例
		# ax.legend(loc='upper right', framealpha=0.8)
	ax.set_title(title, fontsize=18, pad=10)
	ax.set_ylabel('高度 (km)', fontsize=14)
	ax.set_ylim(0, 18)
	ax.tick_params(axis='both', which='major', labelsize=14)
	cbar = fig.colorbar(c, ax=ax, label='dBZ', extend='min', ticks=bounds,pad=0.01)
						# orientation='horizontal',location='bottom',
						# pad=0.15, shrink=0.6, aspect=70)
	cbar.ax.tick_params(labelsize=12)
	cbar.set_label('dBZ', size=14)
	cbar.set_ticklabels([f'{b:.0f}' for b in bounds])
	tick_labels = cbar.ax.get_yticklabels()
	# tick_labels[0].set_text('')
	cbar.ax.set_yticklabels(tick_labels)
def plot_qc_panel(fig, ax, qc_data, timestamps, height_levels, title):
	rcParams['font.sans-serif'] = ['SimHei']
	rcParams['axes.unicode_minus'] = False
	colors = ['black', 'yellow', 'blue']
	bounds = [-1.5, -0.5, 0.5, 1.5]
	cmap = ListedColormap(colors)
	norm = BoundaryNorm(bounds, cmap.N)
	qc_data_for_plot = qc_data.copy().astype(float)
	qc_data_for_plot[qc_data_for_plot == 2] = np.nan
	cmap.set_bad('white', alpha=1)
	c = ax.pcolormesh(timestamps, height_levels, qc_data_for_plot.T, cmap=cmap, norm=norm, shading='auto')
	ax.set_title(title, fontsize=18, pad=10)
	ax.set_ylabel('高度 (km)', fontsize=14)
	ax.set_ylim(0, 20)
	ax.tick_params(axis='both', which='major', labelsize=12)
	cbar = fig.colorbar(c, ax=ax, ticks=[-1, 0, 1])
	cbar.ax.set_yticklabels(['无标签', '晴空(0)', '气象(1)'], fontsize=12)
def create_comparison_plot(results_df, station_info, output_path, plots_to_show):
	# print(f"Generating plot with panels: {plots_to_show}")
	if results_df.empty: return
	draw_ech = False
	if 'ECH_curve' in plots_to_show:
		draw_ech = True
		# 避免后续逻辑将其当做子图去寻找数据列
		plots_to_show = [p for p in plots_to_show if p != 'ECH_curve']
	timestamps = results_df['timestamp'].tolist()
	plot_sources = {
		'original': ('original_z', '质控前', plot_z_panel),
		'human_label_map': ('human_label', '标签值', plot_qc_panel),
		'human_label': ('cleaned_z_human', '人工勾图', plot_z_panel),
		'model_only': ('cleaned_z_model_only', '模型', plot_z_panel),
		'final': ('cleaned_z_final', '质控后', plot_z_panel)
	}
	active_plots = {key: plot_sources[key] for key in plots_to_show if key in plot_sources}
	num_plots = len(active_plots)
	if num_plots == 0: return
	fig, axes = plt.subplots(num_plots, 1, figsize=(16, 6 * num_plots),sharex=False, sharey=True)
	if num_plots == 1: axes = [axes]
	fig.suptitle(
		f"{station_info['code']} ({station_info.get('name', '')}) | {timestamps[0].strftime('%Y-%m-%d')}",
		fontsize=22, y=0.98, x=0.45)
	_, num_height_bins = np.stack(results_df['original_z'].values).shape
	max_hgt = num_height_bins * station_info.get('dis_slt', 30) / 1000.0
	height_levels = np.linspace(0, max_hgt, num_height_bins)
	for i, (key, (data_col, title, plot_func)) in enumerate(active_plots.items()):
		if data_col not in results_df.columns or results_df[data_col].isnull().all():
			axes[i].text(0.5, 0.5, f'无可用数据\n({title})', ha='center', va='center', fontsize=20,
						 transform=axes[i].transAxes)
			axes[i].set_title(title, fontsize=18, pad=10)
			axes[i].set_ylabel('高度 (km)', fontsize=14)
			continue
		data_to_plot = np.stack(results_df.dropna(subset=[data_col])[data_col].values)
		plot_timestamps = results_df.dropna(subset=[data_col])['timestamp'].tolist()
		if 'z' in data_col:
			data_to_plot = data_to_plot.copy().astype(float)
			data_to_plot[data_to_plot <= -99.0] = np.nan
			# 2. 准备 ECH 数据：在这里统一转为 km
			ech_km = None
		# 逻辑：只在“模型预测”或“最终结果”图上叠加 ECH，不在“人工标签”或“原始图”上叠加
		if draw_ech and key in ['final', 'model_only']:
			if 'ech_bins' in results_df.columns:
				# 取出距离库索引
				ech_bins = results_df['ech_bins'].values
				# 统一转换：索引 * (库长/1000)
				ech_km = ech_bins * station_info.get('dis_slt', 30) / 1000.0
		# 3. 传入绘图函数
		if plot_func == plot_z_panel:
			plot_func(fig, axes[i], data_to_plot, plot_timestamps, height_levels, title=title,ech_values=ech_km)
		else:
			plot_func(fig, axes[i], data_to_plot, plot_timestamps, height_levels, title=title)
	for ax in axes:
		ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
		ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
	axes[-1].set_xlabel('时间（UTC）', fontsize=14)
	plt.subplots_adjust(left=0.08, right=0.94, top=0.9, bottom=0.06, hspace=0.25)
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	# print(f"Comparison plot saved to {output_path}")
	plt.close(fig)

def get_ech_ground_truth(label_mask, search_height=200):
	"""
	根据人工标签计算 ECH 真值 (与训练逻辑一致: Class 0 的 98% 分位数)。
	Args:
		label_mask (np.ndarray): (T, H) 的标签矩阵
		search_height (int): 搜索上限 (距离库)，默认 200 (约6km)
	Returns:
		np.ndarray: (T,) 的 ECH 真值序列 (距离库索引)，无效值为 NaN
	"""
	T, H = label_mask.shape
	h_true = np.full(T, np.nan)

	for t in range(T):
		# 获取当前时刻低空范围内的非气象回波 (Class 0) 索引
		# label_mask: 0=Clutter, 1=Weather, 2=Invalid
		# 注意: 需确保只在有效数据范围内统计
		profile = label_mask[t, :search_height]
		clutter_indices = np.where(profile == 0)[0]

		if len(clutter_indices) > 5:  # 设置一个最小像素阈值，避免噪声干扰
			# 计算 98% 分位数
			h_true[t] = np.percentile(clutter_indices, 98)
		else:
			# 如果没有杂波或杂波太少，则认为该时刻无 ECH (或为 0)
			# 这里置为 0 表示边界层极低，或 NaN 表示无法确定
			# 为了计算误差，若无杂波通常意味着 ECH 很低，这里取 0 可能更合适，
			# 但为了严谨，若无杂波则不参与 ECH 误差计算 (设为 NaN)
			h_true[t] = np.nan
	return h_true
# --- 3. 评估函数---
def calculate_metrics(pred_mask, label_mask):
	"""
	计算晴空回波和气象回波的 recall 和 precision。
	"""
	valid_mask = (label_mask < 2)
	if not np.any(valid_mask):
		return np.nan, np.nan, np.nan, np.nan, 0, 0
	# --- 晴空回波 (Clutter, Class 0) ---
	true_clutter = (label_mask == 0) & valid_mask
	pred_clutter = (pred_mask == 0) & valid_mask
	tp_clutter = np.sum(true_clutter & pred_clutter)
	# Recall for Clutter
	total_true_clutter = np.sum(true_clutter)
	clutter_recall = tp_clutter / total_true_clutter if total_true_clutter > 0 else np.nan
	# Precision for Clutter
	total_pred_clutter = np.sum(pred_clutter)
	clutter_precision = tp_clutter / total_pred_clutter if total_pred_clutter > 0 else np.nan
	# --- 气象回波 (Weather, Class 1) ---
	true_weather = (label_mask == 1) & valid_mask
	pred_weather = (pred_mask == 1) & valid_mask
	tp_weather = np.sum(true_weather & pred_weather)
	# Recall for Weather
	total_true_weather = np.sum(true_weather)
	weather_recall = tp_weather / total_true_weather if total_true_weather > 0 else np.nan
	# Precision for Weather
	total_pred_weather = np.sum(pred_weather)
	weather_precision = tp_weather / total_pred_weather if total_pred_weather > 0 else np.nan
	return clutter_recall, weather_recall, clutter_precision, weather_precision, total_true_clutter, total_true_weather

def calculate_comprehensive_metrics(model_qc_mask, final_qc_mask, label_mask,
									h_pred_bins, h_true_bins, station_info):
	"""
	计算综合评估指标：包含全局、分高度层、ECH误差及后处理增益。
	"""
	metrics = {}

	# 基础参数
	range_gate = station_info.get('dis_slt', 30)
	split_height_km = 2.0  # 分层高度: 2km
	split_bin = int(split_height_km * 1000 / range_gate)

	# --- 辅助函数: 计算某一层/区域的 Recall/Precision ---
	def _calc_layer_metrics(pred, label, region_mask, prefix):
		# 仅关注有效区域 (Label != 2) AND 指定区域 (Low/High)
		valid_region = (label < 2) & region_mask

		if not np.any(valid_region):
			return {
				f'{prefix}_C0_Recall': np.nan, f'{prefix}_C0_Prec': np.nan,
				f'{prefix}_C1_Recall': np.nan, f'{prefix}_C1_Prec': np.nan
			}

		# Class 0 (Clutter)
		true_c0 = (label == 0) & valid_region
		pred_c0 = (pred == 0) & valid_region
		tp_c0 = np.sum(true_c0 & pred_c0)
		total_true_c0 = np.sum(true_c0)
		total_pred_c0 = np.sum(pred_c0)  # 新增: 预测为杂波的总数

		# Class 1 (Weather)
		true_c1 = (label == 1) & valid_region
		pred_c1 = (pred == 1) & valid_region
		tp_c1 = np.sum(true_c1 & pred_c1)
		total_true_c1 = np.sum(true_c1)
		total_pred_c1 = np.sum(pred_c1)

		return {
			f'{prefix}_C0_Recall': tp_c0 / total_true_c0 if total_true_c0 > 0 else np.nan,
			f'{prefix}_C0_Prec': tp_c0 / total_pred_c0 if total_pred_c0 > 0 else np.nan,  # 新增
			f'{prefix}_C1_Recall': tp_c1 / total_true_c1 if total_true_c1 > 0 else np.nan,
			f'{prefix}_C1_Prec': tp_c1 / total_pred_c1 if total_pred_c1 > 0 else np.nan
		}

	# 1. 全局掩码 (Label有效区)
	T, H = label_mask.shape
	# 构造高度掩码
	indices = np.arange(H)
	mask_low = np.tile(indices < split_bin, (T, 1))
	mask_high = np.tile(indices >= split_bin, (T, 1))

	# --- 指标 A: 全局 & 分层指标 (针对 Final 结果) ---
	# 全局
	metrics.update(_calc_layer_metrics(final_qc_mask, label_mask, np.ones_like(label_mask, dtype=bool), 'Global'))
	# 低空 (0-2km)
	metrics.update(_calc_layer_metrics(final_qc_mask, label_mask, mask_low, 'Low'))
	# 高空 (>2km)
	metrics.update(_calc_layer_metrics(final_qc_mask, label_mask, mask_high, 'High'))

	# --- 指标 B: ECH 预测误差 ---
	# 过滤掉 NaN 值 (即必须两者都有值才计算误差)
	valid_ech = (~np.isnan(h_pred_bins)) & (~np.isnan(h_true_bins))
	if np.any(valid_ech):
		diff_bins = np.abs(h_pred_bins[valid_ech] - h_true_bins[valid_ech])
		mae_bins = np.mean(diff_bins)
		mae_km = mae_bins * range_gate / 1000.0
		rmse_bins = np.sqrt(np.mean(diff_bins ** 2))
		metrics['ECH_MAE_Bins'] = mae_bins
		metrics['ECH_MAE_Km'] = mae_km
		metrics['ECH_RMSE_Bins'] = rmse_bins
	else:
		metrics['ECH_MAE_Bins'] = np.nan
		metrics['ECH_MAE_Km'] = np.nan
		metrics['ECH_RMSE_Bins'] = np.nan

	# --- 指标 C: 后处理增益 (Post-processing Gain) ---
	# 关注从 Model 到 Final 的变化
	# 有效区域: Label 有效
	valid_mask = label_mask < 2

	# 1. 正确剔除 (Correct Correction): 模型说是 C1 (气象), 后处理改成 C0 (杂波/被滤除), 标签确实是 C0
	# 注意: final_qc_mask 中被滤除通常是变为 0 或 2(NaN), 这里假设滤除后归为非气象(0或2均视为被剔除)
	# 我们主要看: Model=1 AND Final!=1 AND Label=0
	mask_model_c1 = (model_qc_mask == 1)
	mask_final_not_c1 = (final_qc_mask != 1)  # 被后处理移除了
	mask_label_c0 = (label_mask == 0)

	correctly_removed = np.sum(mask_model_c1 & mask_final_not_c1 & mask_label_c0 & valid_mask)

	# 2. 误伤 (Incorrect Removal / Signal Loss): 模型说是 C1, 后处理改成 C0, 但标签其实是 C1
	mask_label_c1 = (label_mask == 1)
	incorrectly_removed = np.sum(mask_model_c1 & mask_final_not_c1 & mask_label_c1 & valid_mask)

	metrics['PP_Correctly_Removed_Count'] = correctly_removed
	metrics['PP_Signal_Lost_Count'] = incorrectly_removed

	# 计算比率 (可选)
	# 修正率: 在模型所有误报(False Positive)中，后处理修好了多少？
	# 这需要计算模型的总 FP，稍微复杂，这里先只存绝对数量，或者相对于总像数的比例
	metrics['PP_Gain_Ratio'] = correctly_removed / (incorrectly_removed + 1e-6)  # 收益代价比

	return metrics
def find_bin_file(bin_path, station, time_key):
	"""通配符匹配查找BIN文件，适配所有中间类型。"""
	bin_files = list(bin_path.glob(f"Z_RADA_I_{station}_{time_key}_O_YCCR_*_RAW_MM.BIN"))
	if bin_files:
		return bin_files[0]
	raise FileNotFoundError(f"未找到匹配的BIN文件: station={station}, time={time_key}")
# --- 4. InferencePipeline 类---
class InferencePipeline:
	def __init__(self, config_path, model_name, model_path, radar_normalizer_path, ground_normalizer_path):
		with open(config_path, encoding='utf-8') as f:
			self.config = yaml.safe_load(f)
		# ---加载新架构所需的配置 ---
		model_config = self.config.get('model', {})
		dataset_config = self.config.get('dataset', {})

		self.model_name = model_name.lower()
		self.height_cutoff = dataset_config.get('height', 500)
		self.window_size = dataset_config.get('window_size', 240)
		self.h_true_max = dataset_config.get('h_true_max_bin', 200.0)
		# --- [GEMINI V4 新增] 读取降采样率 ---
		self.resample_rate = dataset_config.get('resample_rate_minutes', 1)

		self.station_info = {}
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print(f"Initializing model: '{self.model_name}'")
		# --- 适配新模型的实例化参数 ---
		if self.model_name == 'baseline':
			self.model = BaselineModel(
				in_ch=model_config.get('in_ch', 5),  # 5通道: Z,V,W,LDR,H_norm
				out_ch=3
			)
		elif self.model_name == 'baseline+surf':
			self.model = BaselineSurfModel(
				in_ch=model_config.get('in_ch', 5),
				out_ch=3,
				physics_input_dim=model_config.get('physics_input_dim', 3),
				physics_embedding_dim=model_config.get('physics_embedding_dim', 16)
			)
		elif self.model_name == 'baseline+surf+lcl':
			self.model = BaselineSurfLCLModel(
				in_ch=model_config.get('in_ch', 5),
				out_ch=3,
				physics_input_dim=model_config.get('physics_input_dim', 3),
				physics_embedding_dim=model_config.get('physics_embedding_dim', 16)
			)
		else:
			raise ValueError(f"配置文件中的未知模型名称: '{self.model_name}'")

		checkpoint = torch.load(model_path, map_location=self.device)
		if 'model_state_dict' in checkpoint:
			self.model.load_state_dict(checkpoint['model_state_dict'])
		else:
			self.model.load_state_dict(checkpoint)

		self.model.to(self.device)
		self.model.eval()

		self.radar_normalizer = Normalizer(feature_order=['Z', 'V', 'W', 'LDR']).load(radar_normalizer_path)
		# 仅在需要时加载地面归一化器
		if 'surf' in self.model_name:
			self.ground_normalizer = Normalizer(feature_order=['TEM', 'RHU', 'PRE']).load(ground_normalizer_path)
		else:
			self.ground_normalizer = None
		self.data_processor = RadarProcessor(config_path=config_path)
	# --- `_prepare_data_for_range` 适配双输入流 ---
	def _prepare_data_for_range(self, station, start_time, end_time):
			timestamps_to_process = pd.date_range(start=start_time, end=end_time, freq='min')
			# --- [GEMINI V4 修改] 分别为两个输入流创建列表 ---
			all_radar_features, all_physics_features = [], []
			all_original_z, all_human_labels, all_timestamps = [], [], []

			unique_days = timestamps_to_process.to_period('D').unique()
			surf_df_map, label_df_map = {}, {}
			for day in unique_days:
				day_str, month_str = day.strftime("%Y%m%d"), day.strftime("%Y%m")
				next_day_str = (day + 1).strftime("%Y%m%d")
				surf_path = Path(self.config['paths']['raw_surf'].format(station=station, start_time=f"{day_str}000000",
																		 end_time=f"{next_day_str}000000"))
				if surf_path.exists(): surf_df_map[day] = self.data_processor._load_surf(surf_path)
				label_path = Path(self.config['paths']['std_labels'].format(station=station, yearmonth=month_str))
				if label_path.exists() and month_str not in label_df_map:
					label_df_map[month_str] = self.data_processor._load_labels(label_path)

			target_height = self.height_cutoff
			# H_norm (高度) 特征
			h_norm_feature = np.arange(target_height, dtype=np.float32) / target_height

			for timestamp in timestamps_to_process:
				try:
					day_str, time_key = timestamp.strftime("%Y%m%d"), timestamp.strftime("%Y%m%d%H%M%S")
					bin_path_dir = Path(self.config['paths']['raw_radar'].format(station=station, day=day_str))
					bin_path = find_bin_file(bin_path_dir, station, time_key)
					radar_data = self.data_processor.read_bin(bin_path)
					original_z_raw = radar_data.get('Z1')
					if original_z_raw is None or original_z_raw.size == 0: continue

					def standardize_height(arr, pad_value):
						if arr is None: return np.full(target_height, pad_value)
						current_height = arr.shape[0]
						if current_height == target_height:
							return arr
						elif current_height > target_height:
							return arr[:target_height]
						else:
							return np.pad(arr, (0, target_height - current_height), 'constant',
										  constant_values=pad_value)

					original_z = standardize_height(original_z_raw, -100.0)
					v1 = standardize_height(radar_data.get('V1'), -100.0)
					w1 = standardize_height(radar_data.get('W1'), -100.0)
					ldr = standardize_height(radar_data.get('LDR'), -100.0)

					if not self.station_info or self.station_info.get('code') != station:
						self.station_info = {'code': station, 'name': radar_data.get('sta_name', ''),
											 'dis_slt': radar_data.get('dis_slt', 30)}

					# --- [GEMINI V4 修改] 准备雷达输入流 (5通道) ---
					radar_features_list = [original_z, v1, w1, ldr, h_norm_feature]

					# --- [GEMINI V4 修改] 准备物理输入流 (3通道) ---
					physics_features_list = [np.nan, np.nan, np.nan]  # 默认值
					if 'surf' in self.model_name:
						day_period = timestamp.to_period('D')
						if day_period not in surf_df_map: continue
						surf_row = surf_df_map[day_period].loc[surf_df_map[day_period]['Datetime'] == timestamp]
						if surf_row.empty: continue
						# 从 parquet/csv 中读取标量值
						physics_features_list = [
							surf_row['TEM'].values[0],
							surf_row['RHU'].values[0],
							surf_row['PRE'].values[0]
						]

					human_label = np.full(target_height, -1, dtype=np.int64)
					month_str = timestamp.strftime("%Y%m")
					if month_str in label_df_map:
						label_row = label_df_map[month_str].loc[label_df_map[month_str]['timestamp'] == timestamp]
						if not label_row.empty:
							label_from_file = label_row['label'].values[0]
							human_label = standardize_height(label_from_file, -1).astype(np.int64)

					# --- [GEMINI V4 修改] 分别
					all_radar_features.append(np.stack(radar_features_list, axis=-1))  # (H, 5)
					all_physics_features.append(np.array(physics_features_list, dtype=np.float32))  # (3,)
					all_original_z.append(original_z)
					all_human_labels.append(human_label)
					all_timestamps.append(timestamp)
				except FileNotFoundError:
					continue
				except Exception:
					continue

			if not all_radar_features: return None, None, None, None, None

			# --- [GEMINI V4 修改] 返回两个独立的特征序列 ---
			return (np.stack(all_radar_features, axis=0),  # (T, H, 5)
					np.stack(all_physics_features, axis=0),  # (T, 3)
					np.stack(all_original_z, axis=0),  # (T, H)
					np.stack(all_human_labels, axis=0),  # (T, H)
					all_timestamps)
	# --- `_predict_long_sequence` 适配双输入流和降采样 ---
	def _predict_long_sequence(self, radar_features_sequence, physics_features_sequence):
		total_time, _, _ = radar_features_sequence.shape
		# --- 1. 归一化雷达输入流 (5通道) ---
		# 分离 H_norm (第5通道)，因为它不需要归一化
		radar_features_raw = radar_features_sequence[..., 0:4]  # (T, H, 4)
		h_norm_feature = radar_features_sequence[..., 4:5]  # (T, H, 1)
		# 归一化前4个通道
		radar_normalized = self.radar_normalizer.transform(radar_features_raw)
		# 重新组合成 (T, H, 5)
		radar_input_normalized = np.concatenate([radar_normalized, h_norm_feature], axis=-1)

		# --- 2. 归一化物理输入流 (3通道) ---
		physics_input_normalized = np.full_like(physics_features_sequence, 0.0)
		if self.ground_normalizer is not None:
			physics_input_normalized = self.ground_normalizer.transform(physics_features_sequence)  # (T, 3)

		# --- 3. 初始化画布 (与V3相同) ---
		sum_class_logits = np.zeros((total_time, self.height_cutoff, 3), dtype=np.float32)
		sum_lcl_preds = np.zeros(total_time, dtype=np.float32)
		counts = np.zeros(total_time, dtype=np.float32)
		stride = self.window_size // 4

		# --- 4. 滑动窗口推理 ---
		for i in range(0, total_time, stride):
			end_idx = min(i + self.window_size, total_time)
			start_idx = max(0, end_idx - self.window_size)

			# --- 4a. 准备雷达输入张量 (T, H, 5) ---
			window_radar_features_raw = radar_input_normalized[start_idx:end_idx]
			window_radar_features = window_radar_features_raw
			# 窗口末端补零
			if window_radar_features_raw.shape[0] < self.window_size:
				pad_width_rad = ((0, self.window_size - window_radar_features_raw.shape[0]), (0, 0), (0, 0))
				window_radar_features = np.pad(window_radar_features_raw, pad_width_rad, 'constant', constant_values=0)

			radar_input_tensor = torch.from_numpy(window_radar_features).unsqueeze(0).to(self.device, non_blocking=True)

			# --- 4b. 准备物理输入张量 (T_downsampled, 3) ---
			physics_input_tensor = None
			if 'surf' in self.model_name:
				window_physics_features_raw = physics_input_normalized[start_idx:end_idx]

				# --- [GEMINI V4 新增] 在窗口内执行降采样 ---
				resample_rate = self.resample_rate
				target_downsampled_len = self.window_size // resample_rate
				num_timesteps, num_features = window_physics_features_raw.shape
				new_num_timesteps = num_timesteps // resample_rate

				if new_num_timesteps > 0:
					truncated_ts = window_physics_features_raw[:new_num_timesteps * resample_rate]
					downsampled_features = truncated_ts.reshape(new_num_timesteps, resample_rate, num_features).mean(
						axis=1)
				else:  # 处理窗口 < 采样率的边缘情况
					downsampled_features = window_physics_features_raw.mean(axis=0, keepdims=True)

				# 窗口末端补零 (针对降采样后的)
				current_downsampled_len = downsampled_features.shape[0]
				if current_downsampled_len < target_downsampled_len:
					pad_width_phys = ((0, target_downsampled_len - current_downsampled_len), (0, 0))
					downsampled_features = np.pad(downsampled_features, pad_width_phys, 'constant', constant_values=0)

				physics_input_tensor = torch.from_numpy(downsampled_features).unsqueeze(0).to(self.device,
																							  non_blocking=True)

			# --- 4c. 模型推理 (适配 V4) ---
			with torch.no_grad():
				if self.model_name == 'baseline+surf+lcl':
					class_logits, h_pred = self.model(radar_input_tensor.to(torch.float32),
													  physics_input_tensor.to(torch.float32))
					# h_pred 是 (B, T), T=window_size
					h_pred_np = h_pred.squeeze(0).cpu().numpy()
				elif self.model_name == 'baseline+surf':
					class_logits = self.model(radar_input_tensor.to(torch.float32),
											  physics_input_tensor.to(torch.float32))
					h_pred_np = np.full(self.window_size, np.nan, dtype=np.float32)
				elif self.model_name == 'baseline':
					class_logits = self.model(radar_input_tensor.to(torch.float32))
					h_pred_np = np.full(self.window_size, np.nan, dtype=np.float32)

			class_logits_np = class_logits.squeeze(0).cpu().numpy()

			# --- 4d. 累加结果 (与V3相同, h_pred 维度 T 未变) ---
			valid_len = end_idx - start_idx  # 原始 T 维度
			sum_class_logits[start_idx:end_idx] += class_logits_np[:valid_len]
			sum_lcl_preds[start_idx:end_idx] += h_pred_np[:valid_len]
			counts[start_idx:end_idx] += 1.0
			if end_idx == total_time: break

		# --- 5. 平均 (与V3相同) ---
		counts[counts == 0] = 1
		avg_class_logits = sum_class_logits / np.expand_dims(counts, axis=(1, 2))
		avg_lcl_preds = sum_lcl_preds / counts
		final_preds = np.argmax(avg_class_logits, axis=-1)
		return final_preds, avg_lcl_preds
		# ---`process_day` 适配双输入流 ---
	def process_day(self, station, start_time, end_time, output_dir, plots_to_show, structure, do_evaluation=True):
		radar_features_seq, physics_features_seq, original_z_seq, human_label_seq, timestamps = \
			self._prepare_data_for_range(station, start_time, end_time)
		if radar_features_seq is None:
			return None

		preds_low_height, h_pred_sequence_norm = \
			self._predict_long_sequence(radar_features_seq, physics_features_seq)

		total_time, full_height = original_z_seq.shape
		model_qc_mask = np.full((total_time, full_height), 2, dtype=np.int64)
		model_qc_mask[:, :self.height_cutoff] = preds_low_height

		cleaned_z_model_only = original_z_seq.copy().astype(float)
		invalid_mask = (model_qc_mask != 1) | (original_z_seq <= -99.0)
		cleaned_z_model_only[invalid_mask] = np.nan

		if not np.all(np.isnan(h_pred_sequence_norm)):
			h_pred_sequence_bins = h_pred_sequence_norm * self.h_true_max
			safety_margin = self.config['inference'].get('safety_margin', 15)
			dynamic_height_cutoff = int(np.nanmax(h_pred_sequence_bins)) + safety_margin if not np.all(
				np.isnan(h_pred_sequence_bins)) else 150
			final_process_height = min(dynamic_height_cutoff, self.height_cutoff)

			cleaned_z_final = post_process_with_dynamic_lcl(
				z_data=cleaned_z_model_only,
				lcl_predictions_bins=h_pred_sequence_bins,
				safety_margin_bins=safety_margin,
				height_cutoff=final_process_height,
				structure_size=structure
			)
		else:
			h_pred_sequence_bins = np.full_like(h_pred_sequence_norm, np.nan)
			cleaned_z_final = cleaned_z_model_only.copy()


		if plots_to_show:
			cleaned_z_human = original_z_seq.copy().astype(float)
			if human_label_seq is not None and human_label_seq.shape == original_z_seq.shape:
				mask_human = (human_label_seq != 1)
				cleaned_z_human[mask_human] = np.nan
			else:
				cleaned_z_human[:] = np.nan

			results_df = pd.DataFrame({
				'timestamp': timestamps, 'original_z': list(original_z_seq), 'human_label': list(human_label_seq),
				'cleaned_z_human': list(cleaned_z_human), 'cleaned_z_model_only': list(cleaned_z_model_only),
				'cleaned_z_final': list(cleaned_z_final),'ech_bins': list(h_pred_sequence_bins)
			})
			plot_output_dir = Path(output_dir) / self.model_name / station
			plot_output_dir.mkdir(exist_ok=True, parents=True)
			plot_filename = plot_output_dir / f'{station}_{pd.to_datetime(start_time).strftime("%Y%m%d")}.png'

			create_comparison_plot(results_df, self.station_info, plot_filename, plots_to_show)
		if do_evaluation:
			metrics = {
				'站号': station, '日期': pd.to_datetime(start_time).strftime('%Y-%m-%d'),
			}
			if np.all(human_label_seq == -1): return metrics  # 无标签则跳过

			# 1. 准备真值和Mask
			h_true_seq = get_ech_ground_truth(human_label_seq, search_height=200)

			# 构建 Final QC Mask (用于旧指标计算)
			final_qc_mask = model_qc_mask.copy()
			# 逻辑: 模型判为气象(1) 但 后处理结果为NaN -> 说明被后处理剔除 -> 视为杂波(0)
			removed_pixels = (model_qc_mask == 1) & np.isnan(cleaned_z_final)
			final_qc_mask[removed_pixels] = 0

			# 2. 计算旧指标 (兼容原来的Excel列)
			# Model 阶段
			m_rec_0, m_rec_1, m_prec_0, m_prec_1, total_0, total_1 = calculate_metrics(model_qc_mask, human_label_seq)
			# Final 阶段
			f_rec_0, f_rec_1, f_prec_0, f_prec_1, _, _ = calculate_metrics(final_qc_mask, human_label_seq)

			metrics.update({
				'晴空回波总数': total_0, '气象回波总数': total_1,
				'晴空识别率_模型': m_rec_0, '气象保留率_模型': m_rec_1,
				'晴空准确率_模型': m_prec_0, '气象准确率_模型': m_prec_1,
				'晴空识别率_最终': f_rec_0, '气象保留率_最终': f_rec_1,
				'晴空准确率_最终': f_prec_0, '气象准确率_最终': f_prec_1
			})

			# 3. 计算新指标 (详细评估)
			detailed_metrics = calculate_comprehensive_metrics(
				model_qc_mask=model_qc_mask,
				final_qc_mask=final_qc_mask,
				label_mask=human_label_seq,
				h_pred_bins=h_pred_sequence_bins,
				h_true_bins=h_true_seq,
				station_info=self.station_info
			)

			# 4. 将新指标映射为中文名 (写入 Excel 必须)
			# 格式: metrics['中文列名'] = detailed_metrics.get('英文Key')
			metrics.update({
				# --- 低空 (0-2km) ---
				'低空_晴空识别率': detailed_metrics.get('Low_C0_Recall'),
				'低空_气象保留率': detailed_metrics.get('Low_C1_Recall'),
				'低空_气象准确率': detailed_metrics.get('Low_C1_Prec'),
				# --- 高空 (>2km) ---
				'高空_晴空识别率': detailed_metrics.get('High_C0_Recall'),
				'高空_气象保留率': detailed_metrics.get('High_C1_Recall'),
				'高空_气象准确率': detailed_metrics.get('High_C1_Prec'),
				# --- ECH 误差 ---
				'ECH误差_库数(MAE)': detailed_metrics.get('ECH_MAE_Bins'),
				'ECH误差_公里(MAE)': detailed_metrics.get('ECH_MAE_Km'),
				'ECH误差_库数(RMSE)': detailed_metrics.get('ECH_RMSE_Bins'),
				# --- 后处理增益 ---
				'后处理_正确剔除数': detailed_metrics.get('PP_Correctly_Removed_Count'),
				'后处理_误删气象数': detailed_metrics.get('PP_Signal_Lost_Count'),
				'后处理_收益比': detailed_metrics.get('PP_Gain_Ratio')
			})

			return metrics
		return None
# 多进程工作函数
def process_station_wrapper(
		station_id, date_list, config_path, model_name, model_path,
        radar_normalizer_path, ground_normalizer_path,
		output_dir, plots_to_show, structure, evaluation_output_dir, do_evaluation
):
	"""
		每个子进程执行的函数，负责处理单个站点的所有日期。
	"""
	try:
		# `InferencePipeline`
		pipeline = InferencePipeline(
			config_path=config_path,
			model_name=model_name,
			model_path=model_path,
			radar_normalizer_path=radar_normalizer_path,
            ground_normalizer_path=ground_normalizer_path
		)
	except Exception as e:
		print(f"[Process for Station {station_id}] Initialization failed: {e}")
		return []

	station_results = []
	progress_bar = tqdm(date_list, desc=f"Station {station_id}",leave=True)
	for date_str in progress_bar:
		START_TIME = f'{date_str} 00:00:00'
		END_TIME = f'{date_str} 23:59:59'
		metrics = pipeline.process_day(
			station=station_id, start_time=START_TIME, end_time=END_TIME,
			output_dir=output_dir, plots_to_show=plots_to_show, structure=structure,
			do_evaluation=do_evaluation
		)
		if metrics:
			station_results.append(metrics)

	if do_evaluation and station_results:
		output_path = Path(evaluation_output_dir) / model_name
		output_path.mkdir(exist_ok=True, parents=True)
		df_station = pd.DataFrame(station_results)

		# [修复代码] process_station_wrapper 函数内的 column_order 定义
		column_order = [
			'站号', '日期', '晴空回波总数', '气象回波总数',
			'晴空识别率_模型', '晴空准确率_模型', '气象保留率_模型', '气象准确率_模型',
			'晴空识别率_最终', '晴空准确率_最终', '气象保留率_最终', '气象准确率_最终',
			# 新指标 - 低空
			'低空_晴空识别率', '低空_气象保留率', '低空_气象准确率',
			# 新指标 - 高空
			'高空_晴空识别率', '高空_气象保留率', '高空_气象准确率',
			# 新指标 - ECH
			'ECH误差_库数(MAE)', 'ECH误差_公里(MAE)', 'ECH误差_库数(RMSE)',
			# 新指标 - 后处理
			'后处理_正确剔除数', '后处理_误删气象数', '后处理_收益比'
		]
		for col in column_order:
			if col not in df_station.columns: df_station[col] = None
		df_station = df_station[column_order]
		station_filename = f"evaluation_report_{station_id}.xlsx"
		df_station.to_excel(output_path / station_filename, index=False)
	return station_results

# 主执行逻辑
if __name__ == '__main__':
	multiprocessing.freeze_support()
	# --- 1. 基本配置 ---
	CONFIG_PATH = '../configs/config.yaml'
	with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	inference_config = config.get('inference', {})
	# 配置推理结果形式
	ENABLE_PLOTTING = inference_config.get('enable_plotting', False)
	ENABLE_EVALUATION = inference_config.get('enable_evaluation', True)
	# 配置推理对象，训练集、验证集不可同时为True.全量推理要都设置为False，通过指定全量station、start_date、end_date进行
	USE_VALIDATION_SET = inference_config.get('use_validation_set', False)
	USE_TRAIN_SET = inference_config.get('use_train_set', False)

	MODEL_NAME_TO_INFER = config['model'].get('name', 'baseline+surf+lcl')
	MODEL_DIR = Path(config['train']['model_save_path']) / MODEL_NAME_TO_INFER
	model_files = glob.glob(str(MODEL_DIR / f"{MODEL_NAME_TO_INFER}.pth"))
	if not model_files:
		raise FileNotFoundError(f"在目录 {MODEL_DIR} 中未找到任何模型文件。")

	MODEL_PATH = max(model_files, key=os.path.getctime)
	print(f"自动选择模型文件: {MODEL_PATH}")

	RADAR_NORMALIZER_PATH = './radar_normalizer.pkl'
	GROUND_NORMALIZER_PATH = './ground_normalizer.pkl'

	OUTPUT_DIR = inference_config.get('output_dir', '../inference_results')
	PLOT_OUTPUT_DIR = Path(OUTPUT_DIR) / 'Pics'
	REPORT_OUTPUT_DIR = Path(OUTPUT_DIR) / 'Reports'
	cpu_cores = inference_config.get('num_workers', 1)
	structure_list = inference_config.get('structure_size', [3, 3])
	structure = tuple(structure_list)
	print(f"形态学开运算滤波器大小：{structure}")
	print(f"ECH安全裕度：{inference_config.get('safety_margin', 15)}")

	# 检查绘图总开关
	if not ENABLE_PLOTTING:
		PLOTS_TO_SHOW = []  # 完全禁用绘图
		print(f"绘图选项已禁用。")
	else:
		# 只有在启用绘图的情况下才检查配置文件
		if 'plots_to_show' in inference_config:
			PLOTS_TO_SHOW = inference_config['plots_to_show']
		else:
			PLOTS_TO_SHOW = [
				'original',  # 质控前
				'human_label',  # 人工质控基准
				'model_only',  # 模型预测结果
				'final',  # 模型预测+后处理结果
				'ECH_curve',  # 模型预测的ECH曲线
			]
		print(f"启用绘图选项: {PLOTS_TO_SHOW}")
	# --- 3. 根据 'use_validation_set' 获取待处理任务列表 ---
	tasks = []
	print(f"是否推理训练集：{USE_TRAIN_SET}，\n是否推理验证集：{USE_VALIDATION_SET}")
	if USE_VALIDATION_SET ^ USE_TRAIN_SET:
		print(f"--- 模式: 从 './datasplit.npz' 文件中读取推理样本 ---")
		split_cache_path = Path("./datasplit.npz")
		if not split_cache_path.exists():
			raise FileNotFoundError(
				f"错误: 'use_validation_set' 为 true 但未找到 '{split_cache_path}'。\n请先运行训练脚本(train.py)来生成该文件。")
		# 区分对训练集推理的特殊情况
		with np.load(split_cache_path, allow_pickle=True) as data:
			if USE_TRAIN_SET:
				val_files = data['train_files']
				PLOT_OUTPUT_DIR = Path(OUTPUT_DIR) / 'Pics' / 'train_set'
				REPORT_OUTPUT_DIR = Path(OUTPUT_DIR) / 'Reports' / 'train_set'
				# ENABLE_EVALUATION = False
				print("*" * 10+"在训练集上推理"+"*" * 10)
			else:
				val_files = data['val_files']
				PLOT_OUTPUT_DIR = Path(OUTPUT_DIR) / 'Pics' / 'val_set'
				REPORT_OUTPUT_DIR = Path(OUTPUT_DIR) / 'Reports' / 'val_set'
		parsed_entries = set()
		for file_path in val_files:
			try:
				p = Path(file_path)
				parts = p.stem.split('_')  # e.g., ['50953', '20250110']
				station_id = parts[0]
				date_str = parts[1]
				parsed_entries.add((station_id, date_str))
			except Exception as e:
				print(f"警告: 无法从文件名解析站号/日期: {file_path}。错误: {e}")
		for station_id, date_str in sorted(list(parsed_entries)):
			tasks.append({'station': station_id, 'date': date_str})
		print(f"从 'datasplit.npz' 加载了 {len(tasks)} 个唯一的 (站-日) 任务。")
	else:
		INFER_STATIONS = inference_config.get('stations', [])
		INFER_START_DATE = inference_config.get('start_date')
		INFER_END_DATE = inference_config.get('end_date')
		PLOT_OUTPUT_DIR = Path(OUTPUT_DIR) / 'Pics' / 'specific'
		REPORT_OUTPUT_DIR = Path(OUTPUT_DIR) / 'Reports' / 'specific'
		print(f"--- 模式: 从 config['inference']['stations'] 配置中读取任务 ---")
		if not INFER_STATIONS:
			print("警告: 'use_validation_set' 为 false 且 'inference.stations' 为空。没有要处理的任务。")
		elif not INFER_START_DATE or not INFER_END_DATE:
			print(
				"警告: 'use_validation_set' 为 false 但 'inference.start_date' 或 'end_date' 未设置。没有要处理的任务。")
		else:
			date_range = pd.date_range(start=INFER_START_DATE, end=INFER_END_DATE, freq='D')
			for station in INFER_STATIONS:
				for date in date_range:
					tasks.append({'station': str(station), 'date': date.strftime('%Y%m%d')})
			print(
				f"为 {len(INFER_STATIONS)} 个站点生成了 {len(tasks)} 个 (站-日) 任务，日期范围: {INFER_START_DATE} 到 {INFER_END_DATE}。")
	# if ENABLE_PLOTTING:
	# 	print(f"将生成样本的对比图像到 {PLOT_OUTPUT_DIR}")
	if ENABLE_EVALUATION:
		print(f"将生成详细的评估报告到 {REPORT_OUTPUT_DIR}")
	# --- 4. 将任务按站点分组 ---
	tasks_by_station = defaultdict(list)
	for task in tasks:
		tasks_by_station[task['station']].append(task['date'])

	num_stations = len(tasks_by_station)
	if num_stations == 0:
		print("没有要处理的任务，脚本退出。")
		exit()

	# --- 5. 设置并启动多进程池 ---
	max_workers = min(cpu_cores, num_stations)
	print(f"\n使用 {max_workers} 个进程为 {num_stations} 个站点启动多进程推理。")
	pool_args = []
	for station_id, date_list in tasks_by_station.items():
		pool_args.append((
			station_id, date_list, CONFIG_PATH, MODEL_NAME_TO_INFER, MODEL_PATH,
			RADAR_NORMALIZER_PATH, GROUND_NORMALIZER_PATH,
			PLOT_OUTPUT_DIR, PLOTS_TO_SHOW, structure, REPORT_OUTPUT_DIR, ENABLE_EVALUATION
		))

	all_results_nested = []
	with multiprocessing.Pool(processes=max_workers) as pool:
		for result in pool.starmap(process_station_wrapper, pool_args):
			if result:
				all_results_nested.append(result)
				

	# --- 6. 汇总报告 ---
	if ENABLE_EVALUATION:
		# 展平嵌套的结果列表
		all_results_flat = []
		for station_results in all_results_nested:
			if station_results:
				all_results_flat.extend(station_results)

		if all_results_flat:
			print("\n" + "=" * 50 + "\nSaving summary evaluation report...\n" + "=" * 50)
			df_summary = pd.DataFrame(all_results_flat)

			# 定义需要转为百分比字符串的列 (包含新旧指标)
			percent_cols = [
				'晴空识别率_模型', '气象保留率_模型', '晴空准确率_模型', '气象准确率_模型',
				'晴空识别率_最终', '气象保留率_最终', '晴空准确率_最终', '气象准确率_最终',
				'低空_晴空识别率', '低空_气象保留率', '低空_气象准确率',
				'高空_晴空识别率', '高空_气象保留率', '高空_气象准确率'
			]

			# 格式化百分比
			existing_percent_cols = [col for col in percent_cols if col in df_summary.columns]
			for col in existing_percent_cols:
				df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce')
				df_summary[col] = df_summary[col].map(lambda x: f'{x:.4f}' if pd.notna(x) else '', na_action='ignore')

			# [核心修改] 保持与单站报告一致的列顺序
			column_order = [
				'站号', '日期', '晴空回波总数', '气象回波总数',
				'晴空识别率_模型', '晴空准确率_模型', '气象保留率_模型', '气象准确率_模型',
				'晴空识别率_最终', '晴空准确率_最终', '气象保留率_最终', '气象准确率_最终',
				'低空_晴空识别率', '低空_气象保留率', '低空_气象准确率',
				'高空_晴空识别率', '高空_气象保留率', '高空_气象准确率',
				'ECH误差_库数(MAE)', 'ECH误差_公里(MAE)', 'ECH误差_库数(RMSE)',
				'后处理_正确剔除数', '后处理_误删气象数', '后处理_收益比'
			]

			# 补全缺失列
			for col in column_order:
				if col not in df_summary.columns: df_summary[col] = None

			df_summary = df_summary[column_order]

			output_path = Path(REPORT_OUTPUT_DIR) / MODEL_NAME_TO_INFER
			output_path.mkdir(parents=True, exist_ok=True)

			if USE_VALIDATION_SET:
				summary_filename = f"evaluation_summary_validation_set.xlsx"
			else:
				summary_filename = f"evaluation_summary_{INFER_START_DATE}_to_{INFER_END_DATE}.xlsx"

			df_summary.to_excel(output_path / summary_filename, index=False)
			print(f"Saved summary report to {output_path / summary_filename}")
		else:
			print("\nNo evaluation results to save.")

	print("\nBatch processing finished.")

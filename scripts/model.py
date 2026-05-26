#-*- coding:utf-8 -*-
# author: Wang jia peng
# create: 2026/1/22 22:25
# -*- coding:utf-8 -*-
# filename: model.py
# [2025-New-Version] 增强版物理引导模型
# - 引入正弦位置编码 (Positional Encoding) 解决垂直梯度消失问题
# - 增强 ConfidenceHead 拟合能力
# - 保持原有的 U-Net + LCL 双输出架构

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 自定义层 ---
class SqueezeLayer(nn.Module):
	def __init__(self, dim=None):
		super(SqueezeLayer, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.squeeze(self.dim) if self.dim is not None else x.squeeze()


class MaskedConv2d(nn.Module):
	"""部分卷积（Partial Convolution）"""

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
		self.conv_mask = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding, bias=False)
		torch.nn.init.constant_(self.conv_mask.weight, 1.0)
		for param in self.conv_mask.parameters():
			param.requires_grad = False

	def forward(self, x, mask):
		x_masked = x * mask.expand_as(x)
		out = self.conv(x_masked)
		norm_factor = self.conv_mask(mask)
		norm_factor = torch.clamp(norm_factor, min=1e-8)
		out = out / norm_factor.expand_as(out)
		if self.conv.stride[0] > 1 or self.conv.stride[1] > 1:
			new_mask = F.max_pool2d(mask, kernel_size=self.conv.stride, stride=self.conv.stride)
		else:
			new_mask = mask
		return out, new_mask


# --- 新增：位置编码模块 ---
class PositionalEncoding(nn.Module):
	"""
	生成标准 Transformer 风格的正弦/余弦位置编码。
	用于将高度索引 (0~H) 映射为高维向量，强制模型关注垂直位置信息。
	"""

	def __init__(self, d_model, max_len=500):
		super().__init__()
		self.d_model = d_model
		# 创建一个足够长的 PE 矩阵
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		# 注册为 buffer (不作为参数更新，但随模型保存)
		# 形状调整为 [1, d_model, 1, max_len] 以便后续通过 grid_sample 或切片使用
		# 这里为了简单，我们存储为 [max_len, d_model]
		self.register_buffer('pe', pe)

	def forward(self, height_dim):
		"""
		返回前 height_dim 个位置的编码
		Output: [1, d_model, 1, height_dim] 方便广播
		"""
		# 取出前 height_dim 个高度的编码
		# self.pe: [max_len, d_model]
		curr_pe = self.pe[:height_dim, :]  # [H, D]
		# 调整形状: [H, D] -> [1, D, 1, H] (以匹配 Batch, Channel, Time, Height)
		return curr_pe.permute(1, 0).unsqueeze(0).unsqueeze(2)
# --- 核心模块 ---
class Pseudo3DBlock(nn.Module):
	def __init__(self, in_ch, out_ch, temp_kernel=3):
		super().__init__()
		self.conv_dh = MaskedConv2d(in_ch, out_ch, kernel_size=(temp_kernel, 3), padding=(temp_kernel // 2, 1))
		self.bn = nn.BatchNorm2d(out_ch)
		self.relu = nn.ReLU(inplace=True)
		self.conv_c = nn.Conv2d(out_ch, out_ch, kernel_size=1)

	def forward(self, x, mask):
		x, mask = self.conv_dh(x, mask)
		x = self.relu(self.bn(x))
		x = self.conv_c(x)
		return x, mask


class PhysicsEncoder(nn.Module):
	"""物理编码器 (1D-CNN)"""

	def __init__(self, input_dim, embedding_dim):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv1d(input_dim, embedding_dim // 2, kernel_size=7, padding=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(embedding_dim // 2),
			nn.Conv1d(embedding_dim // 2, embedding_dim, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(embedding_dim)
		)

	def forward(self, x):
		# x: [Batch, Time, Channels] -> [Batch, Channels, Time]
		x = x.permute(0, 2, 1)
		x = self.encoder(x)
		return x

class ConfidenceHead(nn.Module):
	"""
	升级版置信度头。
	使用三层 MLP 结构增强非线性交互能力，
	让物理特征和高度位置编码能深度融合。
	"""

	def __init__(self, in_channels):
		super().__init__()
		self.head = nn.Sequential(
			# Layer 1: 融合特征
			nn.Conv2d(in_channels, 64, kernel_size=1),
			nn.ReLU(inplace=True),
			# Layer 2: 深度交互
			nn.Conv2d(64, 32, kernel_size=1),
			nn.ReLU(inplace=True),
			# Layer 3: 输出概率
			nn.Conv2d(32, 1, kernel_size=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.head(x)

# --- 1. 定义极简物理模型 (Wrapper) ---
class PhysicsPriorModel(nn.Module):
	"""
	一个只包含物理分支的极简模型。
	结构与 BaselineSurfLCLModel 中的物理分支完全一致，方便后续权重迁移。
	"""

	def __init__(self, physics_input_dim=3, physics_embedding_dim=16, pe_dim=10):
		super().__init__()
		# 1. 物理编码器 (提取 TEM, RHU, PRE 时序特征)
		self.physics_encoder = PhysicsEncoder(physics_input_dim, physics_embedding_dim)
		# 2. 位置编码 (提供高度信息)
		self.pe_dim = pe_dim
		self.pos_encoder = PositionalEncoding(d_model=pe_dim, max_len=600)
		# 3. 概率头 (物理特征 + 高度PE + LCL -> 概率)
		# 输入维度:
		#   physics_embedding_dim (16)
		#   + pe_dim (10)
		#   + 1 (LCL作为显式特征拼接触发了更强的物理关联)
		self.input_dim = physics_embedding_dim + pe_dim + 1  # +1 for LCL
		self.confidence_head = ConfidenceHead(self.input_dim)

	def forward(self, physics_features, lcl, height_dim):
		# physics_features: [B, T, 3]
		# lcl: [B, T]
		B, T, _ = physics_features.shape
		H = height_dim
		# A. 物理特征编码 -> [B, Emb, T]
		phys_emb = self.physics_encoder(physics_features)
		# B. 扩展到高度维度 -> [B, Emb, T, H]
		phys_map = phys_emb.unsqueeze(-1).expand(-1, -1, -1, H)
		# C. 位置编码 -> [B, PE, T, H]
		pe_map = self.pos_encoder(H).to(physics_features.device)
		pe_map = pe_map.expand(B, -1, T, -1)
		# D. LCL 特征注入 -> [B, 1, T, H] (这是画出好图的关键!)
		lcl_map = lcl.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, H)
		# E. 拼接所有特征
		combined_map = torch.cat([phys_map, pe_map, lcl_map], dim=1)
		# F. 预测概率 -> [B, 1, T, H]
		prob_map = self.confidence_head(combined_map).squeeze(1)

		return prob_map,phys_emb  # [B, T, H]
# ==============================================================================
# --- 方案一: baseline ---
# ==============================================================================
class BaselineModel(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		bottleneck_channels = 128
		self.enc1 = Pseudo3DBlock(in_ch, 64)
		self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
		self.mid = Pseudo3DBlock(64, bottleneck_channels)
		self.up = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True)
		self.dec1 = Pseudo3DBlock(bottleneck_channels + 64, 64)
		self.conv_out = nn.Conv2d(64, out_ch, kernel_size=1)

	def forward(self, features):
		initial_mask = (torch.any(features != 0, dim=-1, keepdim=True)).float()
		radar_features = features.permute(0, 3, 1, 2)
		mask = initial_mask.permute(0, 3, 1, 2)

		enc1, mask_enc1 = self.enc1(radar_features, mask)
		enc2 = self.pool(enc1)
		mask_enc2 = self.pool(mask_enc1)
		mid, mask_mid = self.mid(enc2, mask_enc2)

		dec1_up = self.up(mid)
		mask_dec1_up = self.up(mask_mid)
		dec1_cat = torch.cat([dec1_up, enc1], dim=1)
		mask_cat = torch.max(mask_dec1_up, mask_enc1)
		dec1, _ = self.dec1(dec1_cat, mask_cat)

		out_class = self.conv_out(dec1)
		return out_class.permute(0, 2, 3, 1)
# ==============================================================================
# --- 方案二: baseline+surf (With Positional Encoding) ---
# ==============================================================================
class BaselineSurfModel(nn.Module):
	def __init__(self, in_ch, out_ch, physics_input_dim, physics_embedding_dim):
		super().__init__()
		bottleneck_channels = 128

		# 1. 物理编码器
		self.physics_encoder = PhysicsEncoder(physics_input_dim, physics_embedding_dim)

		# 2. 位置编码器 (固定维度为10，足够捕捉垂直变化)
		self.pe_dim = 10
		self.pos_encoder = PositionalEncoding(d_model=self.pe_dim, max_len=600)

		# 3. 置信度头 (输入 = 物理Emb + PE维度)
		# 注意：这里不再是 +1，而是 +self.pe_dim
		self.confidence_head = ConfidenceHead(physics_embedding_dim + self.pe_dim)

		self.enc1 = Pseudo3DBlock(in_ch, 64)
		self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
		self.mid = Pseudo3DBlock(64, bottleneck_channels)
		self.up = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True)
		self.dec1 = Pseudo3DBlock(bottleneck_channels + 64, 64)
		self.conv_out = nn.Conv2d(64, out_ch, kernel_size=1)

	def forward(self, radar_features, physics_features):
		initial_mask = (torch.any(radar_features != 0, dim=-1, keepdim=True)).float()
		B, T_radar, H, C = radar_features.shape

		# --- A. 物理与高度特征生成 ---
		# 1. 物理特征: [B, Emb, T_phys]
		physics_emb = self.physics_encoder(physics_features)
		if physics_emb.shape[-1] != T_radar:
			physics_emb = F.interpolate(physics_emb, size=T_radar, mode='linear', align_corners=False)

		# 扩展物理特征: [B, Emb, T, H]
		physics_map = physics_emb.unsqueeze(-1).expand(-1, -1, -1, H)

		# 2. 高度位置编码: [1, PE_Dim, 1, H]
		pe_map = self.pos_encoder(H).to(radar_features.device)
		# 扩展到 Batch 和 Time: [B, PE_Dim, T, H]
		pe_map = pe_map.expand(B, -1, T_radar, -1)

		# 3. 拼接: [B, Emb + PE_Dim, T, H]
		combined_map = torch.cat([physics_map, pe_map], dim=1)

		# 4. 生成概率图
		clutter_prob_map = self.confidence_head(combined_map)

		# --- B. 后续 U-Net ---
		clutter_prob_map_permuted = clutter_prob_map.permute(0, 2, 3, 1)
		weighted_mask = initial_mask * (1.0 - clutter_prob_map_permuted)

		radar_features = radar_features.permute(0, 3, 1, 2)
		weighted_mask = weighted_mask.permute(0, 3, 1, 2)

		enc1, mask_enc1 = self.enc1(radar_features, weighted_mask)
		enc2 = self.pool(enc1)
		mask_enc2 = self.pool(mask_enc1)
		mid, mask_mid = self.mid(enc2, mask_enc2)
		dec1_up = self.up(mid)
		mask_dec1_up = self.up(mask_mid)
		dec1_cat = torch.cat([dec1_up, enc1], dim=1)
		mask_cat = torch.max(mask_dec1_up, mask_enc1)
		dec1, _ = self.dec1(dec1_cat, mask_cat)
		out_class = self.conv_out(dec1)

		return out_class.permute(0, 2, 3, 1)
# ==============================================================================
# --- 方案三: 论文所用模型，请重点关注：baseline+surf+lcl (With Positional Encoding) ---
# ==============================================================================
class BaselineSurfLCLModel(nn.Module):
	def __init__(self, in_ch, out_ch, physics_input_dim, physics_embedding_dim, pe_dim=10):
		super().__init__()
		bottleneck_channels = 128
		# ============================================================
		# 1. 模块化替换：不再单独定义 encoder，而是直接集成 PCGM/PIPM
		# ============================================================
		self.prior_module = PhysicsPriorModel(
			physics_input_dim=physics_input_dim,
			physics_embedding_dim=physics_embedding_dim,
			pe_dim=pe_dim
		)
		# 2. 主干 U-Net (保持不变)
		self.enc1 = Pseudo3DBlock(in_ch, 64)
		self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
		self.mid = Pseudo3DBlock(64, bottleneck_channels)
		self.up = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True)
		self.dec1 = Pseudo3DBlock(bottleneck_channels + 64, 64)
		self.conv_out = nn.Conv2d(64, out_ch, kernel_size=1)
		# 3. ECH (LCL) 回归分支适配
		# 只要 physics_embedding_dim 没变，这里的定义不需要动
		lcl_in_channels = bottleneck_channels + physics_embedding_dim + 1
		self.lcl_head = nn.Sequential(
			nn.Conv1d(in_channels=lcl_in_channels, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1),
			nn.Sigmoid()
		)
	def forward(self, radar_features, physics_features, lcl):
		# radar_features: [B, T, H, C]
		# physics_features: [B, T, 3]
		# lcl: [B, T]
		initial_mask = (torch.any(radar_features != 0.0, dim=-1, keepdim=True)).float()
		B, T_radar, H, C = radar_features.shape
		# ============================================================
		# A. 调用物理先验模块 (PCGM)
		# ============================================================
		# 直接获取 概率图(prob_map) 和 物理嵌入(phys_emb)
		clutter_prob_map, phys_emb = self.prior_module(physics_features, lcl, H)
		# 这里的 phys_emb 已经是 [B, Emb, T] 形状
		# 如果 T_radar 和 physics 的 T 不一致，可以在这里做插值
		if phys_emb.shape[-1] != T_radar:
			phys_emb = F.interpolate(phys_emb, size=T_radar, mode='linear', align_corners=False)
		# ============================================================
		# B. 门控操作 (Gating)
		# ============================================================
		# clutter_prob_map: [B, T, H] -> [B, T, H, 1]
		clutter_prob_map = clutter_prob_map.unsqueeze(-1)
		# 应用 Soft Gating
		weighted_mask = initial_mask * (1.0 - clutter_prob_map)
		# ============================================================
		# C. U-Net 主干流程 (输入经过加权的 mask)
		# ============================================================
		radar_features = radar_features.permute(0, 3, 1, 2)  # [B, C, T, H]
		weighted_mask = weighted_mask.permute(0, 3, 1, 2)  # [B, 1, T, H]

		enc1, mask_enc1 = self.enc1(radar_features, weighted_mask)
		enc2 = self.pool(enc1)
		mask_enc2 = self.pool(mask_enc1)
		mid, mask_mid = self.mid(enc2, mask_enc2)
		# ============================================================
		# D. ECH 回归分支 (无需修改结构，只需正确拼接)
		# ============================================================
		# 1. 提取 U-Net 瓶颈层特征
		mid_pooled = F.adaptive_avg_pool2d(mid, (None, 1))  # [B, 128, T, 1]
		mid_squeezed = mid_pooled.squeeze(-1)  # [B, 128, T]
		# 2. 准备 LCL 特征 [B, 1, T]
		lcl_expanded = lcl.unsqueeze(1)
		# 3. 拼接：[U-Net特征, 物理嵌入(来自模块), LCL值]
		# 注意：这里的 phys_emb 就是从 self.prior_module 拿出来的那个
		lcl_input = torch.cat([mid_squeezed, phys_emb, lcl_expanded], dim=1)
		# 4. 预测
		h_pred = self.lcl_head(lcl_input)
		h_pred = h_pred.squeeze(1)

		dec1_up = self.up(mid)
		mask_dec1_up = self.up(mask_mid)
		dec1_cat = torch.cat([dec1_up, enc1], dim=1)
		mask_cat = torch.max(mask_dec1_up, mask_enc1)

		dec1, _ = self.dec1(dec1_cat, mask_cat)
		out_class = self.conv_out(dec1)

		return out_class.permute(0, 2, 3, 1), h_pred


# -*- coding:utf-8 -*-
# filename: train.py
# [GEMINI 功能恢复最终版]
# 这个版本的 train.py 完整恢复了您旧版脚本中所有精细化的功能，
# 包括：严谨的指标计算、详细的Tensorboard日志、带性能指标的文件名、早停机制，
# 并与我们的多模型消融实验方案完全兼容。
import torch
import torch.nn as nn
import time
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import create_loaders
from model import BaselineModel, BaselineSurfModel, BaselineSurfLCLModel
import torch.nn.functional as F
from funcs import *
torch.backends.cudnn.benchmark = True
class FocalLoss(nn.Module):
	"""
	Focal Loss (焦点损失)，用于解决类别不平衡问题。
	它是一个动态加权的CrossEntropyLoss。
	"""
	def __init__(self, gamma=2.0, alpha=None, ignore_index=2, reduction='mean', eps=1e-8):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.ignore_index = ignore_index
		self.reduction = reduction
		self.eps = eps
	def forward(self, logits, target):
		"""
		Args:
			logits: 模型的原始输出 (B, C, T, H)
			target: 真实标签 (B, T, H)
		"""
		# 1. 计算标准的 CrossEntropy Loss (但保留每个像素点的值)
		# 注意：Focal Loss 仍然与“静态alpha权重”结合使用
		ce_loss = F.cross_entropy(logits, target,
								  weight=self.alpha,
								  ignore_index=self.ignore_index,
								  reduction='none')
		# 2. 计算模型对“正确类别”的预测概率 (pt)
		log_pt = -ce_loss  # ce_loss 本质上就是 -log(pt)
		pt = torch.exp(log_pt)
		pt = pt.clamp(min=self.eps, max=1.0 - self.eps)  # 增加数值稳定性

		# 3. 计算 Focal Loss 的核心： (1 - pt)^gamma * ce_loss
		focal_loss = (1 - pt) ** self.gamma * ce_loss

		# 4. 根据 ignore_index 进行掩码和归约
		if self.reduction == 'mean':
			# 创建一个掩码，只选择有效的像素点（非ignore_index）
			valid_mask = (target != self.ignore_index)
			# 对有效像素点的Focal Loss求和，再除以有效像素点的数量
			# 这确保了损失的均值只在有效区域内计算
			if valid_mask.sum() > 0:
				focal_loss = (focal_loss * valid_mask).sum() / valid_mask.sum()
			else:
				# 如果整个批次都是无效值，则损失为0
				focal_loss = focal_loss.sum() * 0.0
		elif self.reduction == 'sum':
			focal_loss = focal_loss.sum()

		return focal_loss

def evaluate(model, model_name, val_loader, device, criterion_class, criterion_lcl, lambda_lcl, num_classes=3):
	"""评估函数，根据模型名称处理不同的输入输出。"""
	model.eval()
	total_val_loss, total_class_loss, total_lcl_loss = 0.0, 0.0, 0.0
	confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
	eval_time = datetime.now().strftime("%H:%M:%S")
	with torch.no_grad():
		for batch in tqdm(val_loader, desc=f"[{eval_time}] Evaluating", leave=True):
			radar_features = batch['radar_features'].to(device, non_blocking=True)
			labels = batch['labels'].to(device, non_blocking=True)
			if model_name == 'baseline':
				outputs_class = model(radar_features)
				h_pred = None
			elif model_name == 'baseline+surf':
				physics_features = batch['physics_features'].to(device, non_blocking=True)
				outputs_class = model(radar_features, physics_features)
				h_pred = None
			else:
				physics_features = batch['physics_features'].to(device, non_blocking=True)
				outputs_class, h_pred = model(radar_features, physics_features)
			loss_class = criterion_class(outputs_class.permute(0, 3, 1, 2), labels)
			if not torch.isnan(loss_class): total_class_loss += loss_class.item()
			loss_lcl = torch.tensor(0.0, device=device)
			if h_pred is not None:
				h_true = batch['h_true'].to(device, non_blocking=True)
				valid_h_mask = ~torch.isnan(h_true)
				if valid_h_mask.any():
					loss_lcl = criterion_lcl(h_pred[valid_h_mask], h_true[valid_h_mask])
					if not torch.isnan(loss_lcl): total_lcl_loss += loss_lcl.item()
			# 只有在模型不是'baseline'且有LCL预测时，LCL损失才计入总损失
			lambda_lcl_weight = lambda_lcl if model_name == 'baseline+surf+lcl' else 0.0
			loss = loss_class + lambda_lcl_weight * loss_lcl
			if not torch.isnan(loss): total_val_loss += loss.item()
			preds = torch.argmax(outputs_class, dim=-1)
			valid_mask = (labels < num_classes - 1)  # 只关注0,1区域的预测
			valid_labels = labels[valid_mask]
			valid_preds = preds[valid_mask]
			if valid_labels.numel() > 0:
				indices = valid_labels * num_classes + valid_preds
				cm_updates = torch.bincount(indices, minlength=num_classes ** 2).view(num_classes, num_classes)
				confusion_matrix += cm_updates
	avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
	avg_class_loss = total_class_loss / len(val_loader) if len(val_loader) > 0 else 0
	avg_lcl_loss = total_lcl_loss / len(val_loader) if len(val_loader) > 0 else 0
	return avg_val_loss, avg_class_loss, avg_lcl_loss, confusion_matrix

def calculate_metrics(cm):
	"""根据混淆矩阵计算指标，考虑了所有类别的混淆情况。"""
	cm = cm.cpu().numpy()
	# 类别0: 晴空回波
	tp0 = cm[0, 0]
	fn0 = cm[0, 1] + cm[0, 2] # 实际是0，预测为1或2
	fp0 = cm[1, 0] + cm[2, 0] # 更严谨的逻辑应包含预测为0但实际为2的情况
	precision0 = tp0 / (tp0 + fp0 + 1e-8)
	recall0 = tp0 / (tp0 + fn0 + 1e-8)
	f1_0 = 2 * (precision0 * recall0) / (precision0 + recall0 + 1e-8)
	# 类别1: 气象回波
	tp1 = cm[1, 1];
	fn1 = cm[1, 0] + cm[1, 2] # 实际是1，预测为0或2
	fp1 = cm[0, 1] + cm[2, 1] # 实际是0或2，预测为1
	precision1 = tp1 / (tp1 + fp1 + 1e-8)
	recall1 = tp1 / (tp1 + fn1 + 1e-8)
	f1_1 = 2 * (precision1 * recall1) / (precision1 + recall1 + 1e-8)
	return {
		'class_0': {'precision': precision0.item(), 'recall': recall0.item(), 'f1': f1_0.item()},
		'class_1': {'precision': precision1.item(), 'recall': recall1.item(), 'f1': f1_1.item()}
	}

def train():
	FORCE_REFIT = False
	RESUME_TRAINING = False
	FORCE_RESPLIT_DATA = True
	config_path = "../configs/config.yaml"
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)
	train_config = config['train']
	model_config = config['model']
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"训练开始，当前使用设备: {device}")
	print("Automatic Mixed Precision (AMP) is disabled for this training session.")
	model_name = model_config.get('name', 'baseline+surf+lcl').lower()
	print(f"★ Initializing model: '{model_name}',input channels：{model_config['in_ch']} ★")
	#focal loss的alpha初始权重
	class_weights = torch.tensor([1.15, 1.0, 1.0], device=device)
	print(f"启用 Focal Loss:Gamma=2.0, Alpha (静态权重): {class_weights}")
	if model_name == 'baseline':
		model = BaselineModel(in_ch=model_config['in_ch'], out_ch=3).to(device)
	elif model_name == 'baseline+surf':
		model = BaselineSurfModel(in_ch=model_config['in_ch'], out_ch=3,
								  physics_input_dim=model_config['physics_input_dim'],
								  physics_embedding_dim=model_config['physics_embedding_dim']).to(device)
	elif model_name == 'baseline+surf+lcl':
		model = BaselineSurfLCLModel(in_ch=model_config['in_ch'], out_ch=3,
									 physics_input_dim=model_config['physics_input_dim'],
									 physics_embedding_dim=model_config['physics_embedding_dim']).to(device)
	else:
		raise ValueError(f"Unknown model name '{model_name}' in config file.")

	optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
	criterion_class = FocalLoss(gamma=2.0, alpha=class_weights, ignore_index=2).to(device)
	criterion_lcl = nn.MSELoss()
	local_time = time.localtime(time.time())
	time_stamp = time.strftime("%Y%m%d%H%M", local_time)
	writer = SummaryWriter(log_dir=f"runs/{model_name.replace('+', '_')}@{time_stamp}")

	# 1. 加载 LCL 权重参数
	dynamic_lcl_config = train_config.get('dynamic_lcl', {})
	use_dynamic_lcl = dynamic_lcl_config.get('enable', False)
	clamping_range = dynamic_lcl_config.get('clamping_range', [0.06, 0.3])
	min_ratio = clamping_range[0]
	max_ratio = clamping_range[1]

	if use_dynamic_lcl:
		print(f"Using DYNAMIC LCL CLAMPING: Clamping Range=[{min_ratio}, {max_ratio}]")
	else:
		print(f"Using STATIC LCL weighting: lambda_lcl={0.1}")

	train_loader, val_loader, _ = create_loaders(config_path,
												 val_dates_output_path=train_config['val_dates_output_path'],
												 force_refit=FORCE_REFIT, force_resplit=FORCE_RESPLIT_DATA)
	batchs_train = len(train_loader)
	print(f"数据加载完成，训练集总批次: {batchs_train}，验证集总批次: {len(val_loader)}")
	model_save_path = Path(train_config['model_save_path']) / model_name
	model_save_path.mkdir(parents=True, exist_ok=True)
	# 初始化日志csv
	log_file_path = initialize_csv_log("./logs", model_name)
	# 需要手动指定要加载的模型名称
	checkpoint_path = model_save_path / f"best_{model_name}.pth"
	start_epoch = 0
	best_avg_recall = 0.0
	epochs_no_improve = 0
	if RESUME_TRAINING and checkpoint_path.exists():
		print(f"发现已存在的模型: {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		start_epoch = checkpoint['epoch'] + 1
		best_avg_recall = checkpoint.get('best_avg_recall', 0.0)
		best_avg_precision = checkpoint.get('best_avg_precision', 0.0)
		epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
		print(f"将从 Epoch {start_epoch + 1} 开始训练。Avg_recall:{best_avg_recall:.4f},Avg_precision:{best_avg_precision:.4f}")
	else:
		print("未找到已保存的模型文件或未启用继续训练，将从头开始训练。")
	epochs = train_config.get('epochs', 100)
	for epoch in range(start_epoch, epochs):
		model.train()
		total_class_loss, total_lcl_loss = 0.0, 0.0
		start_epoch_time = time.time()
		train_time = datetime.now().strftime("%H:%M:%S")
		# N = 10
		# start_batch_index = N - 10  # 从崩溃前10个批次开始，以防万一
		# progress_bar = tqdm(train_loader, desc=f"[{train_time}] Epoch {epoch + 1}/{epochs} [Training]", leave=True)
		progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
							desc=f"[{train_time}] Epoch {epoch + 1}/{epochs} [Training]", leave=True)
		# for batch_idx, batch in itertools.islice(progress_bar, start_batch_index, None):
		for batch_idx, batch in progress_bar:
			radar_features = batch['radar_features'].to(device, non_blocking=True);
			labels = batch['labels'].to(device, non_blocking=True)
			
			optimizer.zero_grad()
			if model_name == 'baseline':
				outputs_class = model(radar_features)
				h_pred = None
			elif model_name == 'baseline+surf':
				physics_features = batch['physics_features'].to(device, non_blocking=True)
				outputs_class = model(radar_features, physics_features)
				h_pred = None
			else:
				physics_features = batch['physics_features'].to(device, non_blocking=True)
				outputs_class, h_pred = model(radar_features, physics_features)

			loss_class = criterion_class(outputs_class.permute(0, 3, 1, 2), labels)
			loss_lcl_raw = torch.tensor(0.0, device=device)
			if h_pred is not None:
				h_true = batch['h_true'].to(device, non_blocking=True)
				valid_h_mask = ~torch.isnan(h_true)
				if valid_h_mask.any():
					loss_lcl_raw = criterion_lcl(h_pred[valid_h_mask], h_true[valid_h_mask])

			# lambda_lcl_weight = train_config['lambda_lcl'] if model_name == 'baseline+surf+lcl' else 0.0
			# 仅在启用钳位、模型是lcl模型且lcl损失大于0时才执行
			if use_dynamic_lcl and model_name == 'baseline+surf+lcl' and loss_lcl_raw.item() > 0:
				# 3. 获取 class_loss 的标量值，并分离计算图 (!!! 关键 !!!)
				# 我们不希望钳位操作的边界影响 class_loss 的梯度
				class_loss_detached = loss_class.detach()
				# 4. 计算钳位的上界和下界
				min_bound = min_ratio * class_loss_detached
				max_bound = max_ratio * class_loss_detached
				# 5. 应用钳位 (Clamping)
				# 确保 LCL 损失的最终贡献值被"钳位"在动态范围内
				loss_lcl_final = torch.clamp(loss_lcl_raw, min=min_bound, max=max_bound)

			else:
				# 6. (默认) 使用标准的静态权重 LCL 损失
				loss_lcl_final = loss_lcl_raw if model_name == 'baseline+surf+lcl' else torch.tensor(0.0,device=device)

			loss = loss_class + loss_lcl_final

			# 使用标准的反向传播
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			if not torch.isnan(loss_class): total_class_loss += loss_class.item()
			if not torch.isnan(loss_lcl_raw): total_lcl_loss += loss_lcl_raw.item()
			progress_bar.set_postfix(cls_loss=f'{loss_class.item():.4f}',
									 lcl_loss_raw=f'{loss_lcl_raw.item():.4f}',
									 lcl_loss_final=f'{loss_lcl_final.item():.4f}',
									 total_loss=f'{loss.item():.4f}'
									 )

		avg_train_class_loss = total_class_loss / len(train_loader) if len(train_loader) > 0 else 0
		avg_train_lcl_loss = total_lcl_loss / len(train_loader) if len(train_loader) > 0 else 0

		avg_val_loss, avg_val_class, avg_val_lcl, conf_matrix = evaluate(model, model_name, val_loader, device,
																		 criterion_class, criterion_lcl,
																		 train_config['lambda_lcl'])
		metrics = calculate_metrics(conf_matrix)
		avg_recall = (metrics['class_0']['recall'] + metrics['class_1']['recall']) / 2
		f1_avg = (metrics['class_0']['f1'] + metrics['class_1']['f1']) / 2
		scheduler.step(avg_val_loss)
		avg_precision = (metrics['class_0']['precision'] + metrics['class_1']['precision']) / 2
		# 详细的日志打印
		epoch_duration = time.time() - start_epoch_time
		current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		print("-" * 80)
		print(
			f"[{current_time}] Epoch {epoch + 1}/{epochs} | Time: {epoch_duration:.2f}s | Train Loss: class:{avg_train_class_loss:.4f}/lcl:{avg_train_lcl_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
		print(
			f"  Class 0 - Precision: {metrics['class_0']['precision']:.4f}, Recall: {metrics['class_0']['recall']:.4f}, F1: {metrics['class_0']['f1']:.4f}")
		print(
			f"  Class 1 - Precision: {metrics['class_1']['precision']:.4f}, Recall: {metrics['class_1']['recall']:.4f}, F1: {metrics['class_1']['f1']:.4f}")
		print(f"  Average F1 Score: {f1_avg:.4f} | Average Recall: {avg_recall:.4f}")
		print("-" * 80)

		# 详细的Tensorboard日志记录
		writer.add_scalar('Loss/train_class', avg_train_class_loss, epoch)
		writer.add_scalar('Loss/train_lcl', avg_train_lcl_loss, epoch)
		writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
		writer.add_scalar('Loss/val_class', avg_val_class, epoch)
		writer.add_scalar('Loss/val_lcl', avg_val_lcl, epoch)
		writer.add_scalar('Metrics/F1_Class_0', metrics['class_0']['f1'], epoch)
		writer.add_scalar('Metrics/F1_Class_1', metrics['class_1']['f1'], epoch)
		writer.add_scalar('Metrics/F1_Average', f1_avg, epoch)
		writer.add_scalar('Metrics/Recall_Class_0', metrics['class_0']['recall'], epoch)
		writer.add_scalar('Metrics/Recall_Class_1', metrics['class_1']['recall'], epoch)
		writer.add_scalar('Metrics/Recall_Average', avg_recall, epoch)
		writer.add_scalar('Metrics/Precision_Class_0', metrics['class_0']['precision'], epoch)
		writer.add_scalar('Metrics/Precision_Class_1', metrics['class_1']['precision'], epoch)
		writer.add_scalar('Metrics/Precision_Average', avg_precision, epoch)
		writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

		if avg_recall > best_avg_recall:
			best_avg_recall = avg_recall
			epochs_no_improve = 0
			# 包含性能指标的模型文件名保存逻辑
			recall_0 = f"{metrics['class_0']['recall']:.4f}"
			recall_1 = f"{metrics['class_1']['recall']:.4f}"
			# 清理旧的最佳模型文件
			# for f in model_save_path.glob("best_model_*.pth"): f.unlink()
			save_path = model_save_path / f"{model_name}_recall_{avg_recall:.4f}_r0_{recall_0}_r1_{recall_1}.pth"
			torch.save(
				{'epoch': epoch,
				 'model_state_dict': model.state_dict(),
				 'optimizer_state_dict': optimizer.state_dict(),
				 'scheduler_state_dict': scheduler.state_dict(),
				 'best_avg_recall': best_avg_recall,
				 'best_avg_precision': avg_precision,
				 'epochs_no_improve': epochs_no_improve}, save_path)
			print("*" * 80)
			print(f"★★★ New best model saved to {save_path} ★★★")
			print("*" * 80)
		else:
			epochs_no_improve += 1
		# 准备写入日志的超参数
		lcl_range_str = str(clamping_range) if use_dynamic_lcl else 'N/A'
		lcl_static_str = train_config[
			'lambda_lcl'] if not use_dynamic_lcl and model_name == 'baseline+surf+lcl' else 'N/A'
		phys_in_dim = model_config.get('physics_input_dim', 'N/A') if model_name != 'baseline' else 'N/A'
		phys_emb_dim = model_config.get('physics_embedding_dim', 'N/A') if model_name != 'baseline' else 'N/A'
		alpha_str = str(criterion_class.alpha.cpu().numpy() if criterion_class.alpha is not None else "None")
		log_data = {
			'timestamp': current_time,
			'model_name': model_name,
			'epoch': epoch + 1,
			'learning_rate': optimizer.param_groups[0]['lr'],
			'batch_size': train_config['batch_size'],
			'in_ch': model_config['in_ch'],
			'focalloss_gamma': criterion_class.gamma,
			'focalloss_alpha': alpha_str,
			'use_dynamic_lcl': use_dynamic_lcl,
			'lambda_lcl_static': lcl_static_str,
			'clamping_range': lcl_range_str,
			'physics_input_dim': phys_in_dim,
			'physics_embedding_dim': phys_emb_dim,
			'train_class_loss': round(avg_train_class_loss, 4),
			'train_lcl_loss': round(avg_train_lcl_loss, 4),
			'val_total_loss': round(avg_val_loss, 4),
			'val_class_loss': round(avg_val_class, 4),
			'val_lcl_loss': round(avg_val_lcl, 4),
			'class0_precision': round(metrics['class_0']['precision'], 4),
			'class0_recall': round(metrics['class_0']['recall'], 4),
			'class0_f1': round(metrics['class_0']['f1'], 4),
			'class1_precision': round(metrics['class_1']['precision'], 4),
			'class1_recall': round(metrics['class_1']['recall'], 4),
			'class1_f1': round(metrics['class_1']['f1'], 4),
			'avg_precision': round(avg_precision, 4),
			'avg_recall': round(avg_recall, 4),
			'avg_f1': round(f1_avg, 4),
			'epoch_duration_s': f"{epoch_duration:.2f}",
		}
		append_epoch_log(log_file_path, log_data)
		if epochs_no_improve >= train_config.get('early_stopping_patience', 10):
			print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement in average recall.")
			break

if __name__ == '__main__':
	train()
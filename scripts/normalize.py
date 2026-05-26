# filename: normalize.py
# 这个版本的 Normalizer 被设计为更通用的形式，可以独立处理不同来源和维度的数据。

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class InvalidValueTransformer(BaseEstimator, TransformerMixin):
    """一个将指定无效值转换为np.nan的sklearn兼容转换器"""
    def __init__(self, invalid_value=-100.0):
        self.invalid_value = invalid_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy().astype(np.float32)
        X_copy[X_copy == self.invalid_value] = np.nan
        return X_copy

class Normalizer:
    """
    一个通用的归一化类，封装了数据预处理流程。
    可以被多次实例化，以处理不同类型的数据流（例如，雷达数据和地面数据）。
    """
    def __init__(self, feature_order):
        """
        构造函数。
        Args:
            feature_order (list): 一个字符串列表，定义了该归一化器要处理的特征及其顺序。
        """
        self.feature_order = feature_order
        self.invalid_value = -100.0

        # 定义一个标准的预处理流水线
        self.pipeline = Pipeline([
            # 步骤1: 将指定的无效值（如-100.0）转换为空值（NaN）
            ('invalid_to_nan', InvalidValueTransformer(invalid_value=self.invalid_value)),
            # 步骤2: 标准化，将数据缩放到均值为0，方差为1
            ('scaler', StandardScaler()),
            # 步骤3: 填充空值，用0.0填充所有剩余的NaN值
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0))
        ])

    def fit(self, data):
        """
        使用提供的数据来拟合（计算均值和标准差）归一化器。
        Args:
            data (np.ndarray): 一个二维Numpy数组，形状为 [样本数, 特征数]，用于训练。
        """
        print(f"Fitting Normalizer for features: {self.feature_order}...")
        self.pipeline.fit(data)
        scaler = self.pipeline.named_steps['scaler']
        print(f"Fitting complete. Mean: {scaler.mean_}, Std: {np.sqrt(scaler.var_)}")
        return self

    def transform(self, data):
        """
        使用已经拟合好的统计量来转换新的数据。
        Args:
            data (np.ndarray): 任意维度的Numpy数组，最后一维是特征维度。
        Returns:
            np.ndarray: 经过归一化处理后的数据，形状与输入相同。
        """
        # 记录原始形状
        original_shape = data.shape
        # 获取特征维度
        num_features = original_shape[-1]

        # 检查特征维度是否匹配
        if num_features != len(self.feature_order):
            raise ValueError(f"Input data has {num_features} features, but normalizer was fitted for {len(self.feature_order)} features.")

        # 将数据展平为二维 [样本数, 特征数] 以进行处理
        data_2d = data.reshape(-1, num_features)
        # 应用归一化流程
        transformed_2d = self.pipeline.transform(data_2d)
        # 将数据恢复为原始形状
        return transformed_2d.reshape(original_shape)

    def save(self, filepath):
        """将拟合好的流水线保存到文件。"""
        joblib.dump(self.pipeline, filepath)
        print(f"Normalizer for {self.feature_order} saved to {filepath}")

    def load(self, filepath):
        """从文件加载已经拟合好的流水线。"""
        self.pipeline = joblib.load(filepath)
        print(f"Normalizer for {self.feature_order} loaded from {filepath}")
        return self


"""
特征工程模块
负责创建和转换特征，为机器学习模型准备数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import holidays

from config.settings import FEATURE_CONFIG, NYC_BOUNDS
from config.logging_config import LoggerMixin, log_execution_time

warnings.filterwarnings('ignore')

class FeatureEngineer(LoggerMixin):
    """
    特征工程类
    提供全面的特征创建和转换功能
    """
    
    def __init__(self):
        """初始化特征工程器"""
        super().__init__()
        self.log_info("FeatureEngineer initialized")
        self.scalers = {}
        self.encoders = {}
        self.pca_models = {}
        self.cluster_models = {}
    
    @log_execution_time
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            df: 原始DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        self.log_info("Creating all features")
        
        # 复制数据避免修改原始数据
        feature_df = df.copy()
        
        # 1. 时间特征
        feature_df = self.create_time_features(feature_df)
        
        # 2. 空间特征
        feature_df = self.create_spatial_features(feature_df)
        
        # 3. 业务特征
        feature_df = self.create_business_features(feature_df)
        
        # 4. 交互特征
        feature_df = self.create_interaction_features(feature_df)
        
        # 5. 统计特征
        feature_df = self.create_statistical_features(feature_df)
        
        # 6. 滞后特征
        feature_df = self.create_lag_features(feature_df)
        
        # 7. 滚动特征
        feature_df = self.create_rolling_features(feature_df)
        
        self.log_info(f"Feature engineering completed. Final shape: {feature_df.shape}")
        
        return feature_df
    
    @log_execution_time
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建时间特征
        
        Args:
            df: DataFrame
            
        Returns:
            包含时间特征的DataFrame
        """
        self.log_info("Creating time features")
        
        time_col = 'datetime' if 'datetime' in df.columns else 'Date/Time'
        
        if time_col not in df.columns:
            self.log_warning(f"Time column {time_col} not found")
            return df
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # 基础时间特征
        df['hour'] = df[time_col].dt.hour
        df['day'] = df[time_col].dt.day
        df['weekday'] = df[time_col].dt.weekday
        df['month'] = df[time_col].dt.month
        df['quarter'] = df[time_col].dt.quarter
        df['year'] = df[time_col].dt.year
        df['day_of_year'] = df[time_col].dt.dayofyear
        df['week_of_year'] = df[time_col].dt.isocalendar().week
        
        # 周期性特征
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 业务时间特征
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['is_peak_hour'] = self._is_peak_hour(df['hour']).astype(int)
        df['is_morning'] = df['hour'].isin([6, 7, 8, 9]).astype(int)
        df['is_afternoon'] = df['hour'].isin([12, 13, 14, 15, 16]).astype(int)
        df['is_evening'] = df['hour'].isin([17, 18, 19, 20]).astype(int)
        df['is_night'] = df['hour'].isin([21, 22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # 节假日特征
        df['is_holiday'] = self._is_holiday(df[time_col]).astype(int)
        
        # 时间间隔特征
        df['time_since_midnight'] = df[time_col].dt.hour * 3600 + df[time_col].dt.minute * 60 + df[time_col].dt.second
        df['time_since_week_start'] = df['weekday'] * 24 * 3600 + df['time_since_midnight']
        
        self.log_info("Time features created successfully")
        return df
    
    def _is_peak_hour(self, hour_series: pd.Series) -> pd.Series:
        """判断是否为高峰时段"""
        return (hour_series.isin([7, 8, 9]) | hour_series.isin([17, 18, 19]))
    
    def _is_holiday(self, datetime_series: pd.Series) -> pd.Series:
        """判断是否为节假日"""
        try:
            us_holidays = holidays.US()
            return datetime_series.dt.date.apply(lambda x: x in us_holidays)
        except:
            # 如果holidays库不可用，返回全False
            return pd.Series([False] * len(datetime_series), index=datetime_series.index)
    
    @log_execution_time
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建空间特征
        
        Args:
            df: DataFrame
            
        Returns:
            包含空间特征的DataFrame
        """
        self.log_info("Creating spatial features")
        
        lat_col = 'latitude' if 'latitude' in df.columns else 'Lat'
        lon_col = 'longitude' if 'longitude' in df.columns else 'Lon'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            self.log_warning("Latitude or longitude columns not found")
            return df
        
        # 距离特征
        nyc_center_lat, nyc_center_lon = 40.7128, -74.0060  # NYC中心坐标
        
        df['distance_to_center'] = self._calculate_distance(
            df[lat_col], df[lon_col], nyc_center_lat, nyc_center_lon
        )
        
        # 区域编码（网格化）
        df['grid_lat'] = ((df[lat_col] - NYC_BOUNDS['min_lat']) / 0.01).astype(int)
        df['grid_lon'] = ((df[lon_col] - NYC_BOUNDS['min_lon']) / 0.01).astype(int)
        df['grid_id'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)
        
        # 聚类特征
        df = self._add_clustering_features(df, lat_col, lon_col)
        
        # 密度特征
        df = self._add_density_features(df, lat_col, lon_col)
        
        self.log_info("Spatial features created successfully")
        return df
    
    def _calculate_distance(self, lat1: pd.Series, lon1: pd.Series, 
                          lat2: float, lon2: float) -> pd.Series:
        """计算距离（Haversine公式）"""
        R = 6371  # 地球半径（公里）
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _add_clustering_features(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
        """添加聚类特征"""
        try:
            # 使用K-means聚类
            coords = df[[lat_col, lon_col]].values
            
            # 尝试不同的聚类数
            for n_clusters in [5, 8, 10]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(coords)
                df[f'cluster_{n_clusters}'] = cluster_labels
                
                # 保存聚类模型
                self.cluster_models[f'kmeans_{n_clusters}'] = kmeans
            
            self.log_info("Clustering features added successfully")
            
        except Exception as e:
            self.log_error(f"Error in clustering: {str(e)}")
        
        return df
    
    def _add_density_features(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
        """添加密度特征"""
        try:
            # 计算每个点周围的点数（简单密度估计）
            from scipy.spatial.distance import cdist
            
            coords = df[[lat_col, lon_col]].values
            
            # 计算每个点1km范围内的点数
            distances = cdist(coords, coords)
            density_1km = (distances < 0.01).sum(axis=1) - 1  # 减去自己
            df['density_1km'] = density_1km
            
            # 计算每个点5km范围内的点数
            density_5km = (distances < 0.05).sum(axis=1) - 1
            df['density_5km'] = density_5km
            
        except Exception as e:
            self.log_error(f"Error in density calculation: {str(e)}")
        
        return df
    
    @log_execution_time
    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建业务特征
        
        Args:
            df: DataFrame
            
        Returns:
            包含业务特征的DataFrame
        """
        self.log_info("Creating business features")
        
        # Base站特征
        base_col = 'base_station' if 'base_station' in df.columns else 'Base'
        if base_col in df.columns:
            # Base站订单数量
            base_counts = df[base_col].value_counts()
            df['base_order_count'] = df[base_col].map(base_counts)
            
            # Base站编码
            if base_col not in self.encoders:
                self.encoders[base_col] = LabelEncoder()
                df[f'{base_col}_encoded'] = self.encoders[base_col].fit_transform(df[base_col])
            else:
                df[f'{base_col}_encoded'] = self.encoders[base_col].transform(df[base_col])
        
        # 需求预测特征
        df['demand_forecast'] = self._calculate_demand_forecast(df)
        
        # 供需平衡特征
        df['supply_demand_ratio'] = self._calculate_supply_demand_ratio(df)
        
        # 价格特征（如果有的话）
        df = self._add_price_features(df)
        
        self.log_info("Business features created successfully")
        return df
    
    def _calculate_demand_forecast(self, df: pd.DataFrame) -> pd.Series:
        """计算需求预测"""
        # 基于历史数据的简单预测
        # 这里可以根据实际业务逻辑进行更复杂的计算
        
        # 按小时和星期的平均需求
        hourly_weekday_avg = df.groupby(['hour', 'weekday']).size().reset_index(name='avg_demand')
        hourly_weekday_avg = hourly_weekday_avg.set_index(['hour', 'weekday'])['avg_demand']
        
        # 为每个记录分配预测值
        forecast = df.apply(
            lambda row: hourly_weekday_avg.get((row['hour'], row['weekday']), 0), 
            axis=1
        )
        
        return forecast
    
    def _calculate_supply_demand_ratio(self, df: pd.DataFrame) -> pd.Series:
        """计算供需比例"""
        # 这里可以根据实际业务逻辑计算
        # 暂时返回随机值作为示例
        return np.random.uniform(0.5, 2.0, len(df))
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格特征"""
        # 如果有价格数据，可以添加价格相关特征
        # 暂时跳过
        return df
    
    @log_execution_time
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建交互特征
        
        Args:
            df: DataFrame
            
        Returns:
            包含交互特征的DataFrame
        """
        self.log_info("Creating interaction features")
        
        # 时间-空间交互
        if 'hour' in df.columns and 'cluster_8' in df.columns:
            df['hour_cluster_interaction'] = df['hour'] * df['cluster_8']
        
        # 时间-业务交互
        if 'hour' in df.columns and 'is_weekend' in df.columns:
            df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']
        
        # 空间-业务交互
        if 'distance_to_center' in df.columns and 'is_peak_hour' in df.columns:
            df['distance_peak_interaction'] = df['distance_to_center'] * df['is_peak_hour']
        
        # 多项式特征
        if 'hour' in df.columns:
            df['hour_squared'] = df['hour'] ** 2
            df['hour_cubed'] = df['hour'] ** 3
        
        if 'distance_to_center' in df.columns:
            df['distance_squared'] = df['distance_to_center'] ** 2
        
        self.log_info("Interaction features created successfully")
        return df
    
    @log_execution_time
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建统计特征
        
        Args:
            df: DataFrame
            
        Returns:
            包含统计特征的DataFrame
        """
        self.log_info("Creating statistical features")
        
        # 按时间窗口的统计特征
        if 'datetime' in df.columns:
            # 按小时统计
            hourly_stats = df.groupby('hour').agg({
                'latitude': ['mean', 'std', 'count'],
                'longitude': ['mean', 'std']
            }).reset_index()
            
            hourly_stats.columns = ['hour', 'lat_mean', 'lat_std', 'order_count', 'lon_mean', 'lon_std']
            
            # 合并回原数据
            df = df.merge(hourly_stats, on='hour', how='left')
        
        # 按Base站统计
        base_col = 'base_station' if 'base_station' in df.columns else 'Base'
        if base_col in df.columns:
            base_stats = df.groupby(base_col).agg({
                'latitude': ['mean', 'std', 'count'],
                'longitude': ['mean', 'std']
            }).reset_index()
            
            base_stats.columns = [base_col, 'base_lat_mean', 'base_lat_std', 'base_order_count', 'base_lon_mean', 'base_lon_std']
            
            df = df.merge(base_stats, on=base_col, how='left')
        
        self.log_info("Statistical features created successfully")
        return df
    
    @log_execution_time
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: DataFrame
            
        Returns:
            包含滞后特征的DataFrame
        """
        self.log_info("Creating lag features")
        
        # 需要按时间排序
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 按小时聚合订单数
            hourly_orders = df.groupby(pd.Grouper(key='datetime', freq='H')).size().reset_index(name='order_count')
            hourly_orders = hourly_orders.set_index('datetime')
            
            # 创建滞后特征
            for lag in [1, 2, 3, 6, 12, 24]:
                hourly_orders[f'order_count_lag_{lag}'] = hourly_orders['order_count'].shift(lag)
            
            # 合并回原数据
            df['datetime_hour'] = df['datetime'].dt.floor('H')
            df = df.merge(hourly_orders.reset_index(), left_on='datetime_hour', right_on='datetime', how='left')
            df = df.drop('datetime_hour', axis=1)
        
        self.log_info("Lag features created successfully")
        return df
    
    @log_execution_time
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建滚动特征
        
        Args:
            df: DataFrame
            
        Returns:
            包含滚动特征的DataFrame
        """
        self.log_info("Creating rolling features")
        
        # 需要按时间排序
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 按小时聚合订单数
            hourly_orders = df.groupby(pd.Grouper(key='datetime', freq='H')).size().reset_index(name='order_count')
            hourly_orders = hourly_orders.set_index('datetime')
            
            # 创建滚动特征
            for window in [3, 6, 12, 24]:
                hourly_orders[f'order_count_rolling_mean_{window}'] = hourly_orders['order_count'].rolling(window=window, min_periods=1).mean()
                hourly_orders[f'order_count_rolling_std_{window}'] = hourly_orders['order_count'].rolling(window=window, min_periods=1).std()
                hourly_orders[f'order_count_rolling_max_{window}'] = hourly_orders['order_count'].rolling(window=window, min_periods=1).max()
                hourly_orders[f'order_count_rolling_min_{window}'] = hourly_orders['order_count'].rolling(window=window, min_periods=1).min()
            
            # 合并回原数据
            df['datetime_hour'] = df['datetime'].dt.floor('H')
            df = df.merge(hourly_orders.reset_index(), left_on='datetime_hour', right_on='datetime', how='left')
            df = df.drop('datetime_hour', axis=1)
        
        self.log_info("Rolling features created successfully")
        return df
    
    @log_execution_time
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        编码分类特征
        
        Args:
            df: DataFrame
            categorical_columns: 分类列名列表
            
        Returns:
            编码后的DataFrame
        """
        self.log_info("Encoding categorical features")
        
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        encoded_df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                # 标签编码
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    encoded_df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].fillna('Unknown'))
                else:
                    encoded_df[f'{col}_encoded'] = self.encoders[col].transform(df[col].fillna('Unknown'))
                
                # One-hot编码（对于低基数分类变量）
                if df[col].nunique() <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                    encoded_df = pd.concat([encoded_df, dummies], axis=1)
        
        self.log_info("Categorical features encoded successfully")
        return encoded_df
    
    @log_execution_time
    def scale_numerical_features(self, df: pd.DataFrame, 
                                numerical_columns: Optional[List[str]] = None,
                                method: str = 'standard') -> pd.DataFrame:
        """
        缩放数值特征
        
        Args:
            df: DataFrame
            numerical_columns: 数值列名列表
            method: 缩放方法 ('standard', 'robust', 'minmax')
            
        Returns:
            缩放后的DataFrame
        """
        self.log_info(f"Scaling numerical features using {method} method")
        
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除不需要缩放的特征
        exclude_cols = ['hour', 'day', 'weekday', 'month', 'quarter', 'year', 
                       'is_weekend', 'is_peak_hour', 'is_holiday', 'cluster_5', 'cluster_8', 'cluster_10']
        numerical_columns = [col for col in numerical_columns if col not in exclude_cols]
        
        scaled_df = df.copy()
        
        for col in numerical_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                scaler_name = f"{method}_{col}"
                
                if method == 'standard':
                    if scaler_name not in self.scalers:
                        self.scalers[scaler_name] = StandardScaler()
                        scaled_df[f'{col}_scaled'] = self.scalers[scaler_name].fit_transform(df[[col]])
                    else:
                        scaled_df[f'{col}_scaled'] = self.scalers[scaler_name].transform(df[[col]])
                
                elif method == 'robust':
                    if scaler_name not in self.scalers:
                        self.scalers[scaler_name] = RobustScaler()
                        scaled_df[f'{col}_scaled'] = self.scalers[scaler_name].fit_transform(df[[col]])
                    else:
                        scaled_df[f'{col}_scaled'] = self.scalers[scaler_name].transform(df[[col]])
        
        self.log_info("Numerical features scaled successfully")
        return scaled_df
    
    def get_feature_importance(self, df: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            df: DataFrame
            target_column: 目标列名
            
        Returns:
            特征重要性字典
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # 准备特征和目标变量
            feature_cols = [col for col in df.columns if col != target_column and not col.startswith('datetime')]
            X = df[feature_cols].fillna(0)
            y = df[target_column]
            
            # 训练随机森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # 获取特征重要性
            importance_dict = dict(zip(feature_cols, rf.feature_importances_))
            
            # 按重要性排序
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            self.log_error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def select_features(self, df: pd.DataFrame, target_column: str, 
                       method: str = 'importance', threshold: float = 0.01) -> List[str]:
        """
        特征选择
        
        Args:
            df: DataFrame
            target_column: 目标列名
            method: 选择方法 ('importance', 'correlation', 'variance')
            threshold: 阈值
            
        Returns:
            选中的特征列表
        """
        self.log_info(f"Selecting features using {method} method")
        
        if method == 'importance':
            importance_dict = self.get_feature_importance(df, target_column)
            selected_features = [feat for feat, imp in importance_dict.items() if imp > threshold]
        
        elif method == 'correlation':
            # 基于相关性的特征选择
            corr_matrix = df.corr()[target_column].abs()
            selected_features = corr_matrix[corr_matrix > threshold].index.tolist()
            selected_features.remove(target_column)
        
        elif method == 'variance':
            # 基于方差的特征选择
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=threshold)
            feature_cols = [col for col in df.columns if col != target_column]
            X = df[feature_cols].fillna(0)
            selector.fit(X)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        else:
            self.log_warning(f"Unknown feature selection method: {method}")
            selected_features = [col for col in df.columns if col != target_column]
        
        self.log_info(f"Selected {len(selected_features)} features")
        return selected_features

# 便捷函数
def create_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    为预测模型创建特征
    
    Args:
        df: 原始DataFrame
        
    Returns:
        包含所有特征的DataFrame
    """
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)

def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    获取特征摘要
    
    Args:
        df: DataFrame
        
    Returns:
        特征摘要字典
    """
    summary = {
        'total_features': len(df.columns),
        'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=['object']).columns),
        'datetime_features': len(df.select_dtypes(include=['datetime64']).columns),
        'feature_types': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return summary 
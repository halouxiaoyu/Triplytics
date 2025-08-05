"""
数据清洗器模块
负责数据清洗、预处理和质量控制
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

from config.settings import (
    NYC_BOUNDS, COLUMN_MAPPING, DATE_FORMAT, TIMEZONE
)
from config.logging_config import LoggerMixin, log_execution_time

warnings.filterwarnings('ignore')

class DataCleaner(LoggerMixin):
    """
    数据清洗器类
    提供全面的数据清洗和预处理功能
    """
    
    def __init__(self):
        """初始化数据清洗器"""
        super().__init__()
        self.log_info("DataCleaner initialized")
    
    @log_execution_time
    def clean_uber_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗Uber数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        self.log_info("Starting Uber data cleaning process")
        
        # 复制数据避免修改原始数据
        cleaned_df = df.copy()
        
        # 1. 标准化列名
        cleaned_df = self.standardize_columns(cleaned_df)
        
        # 2. 处理时间字段
        cleaned_df = self.process_datetime(cleaned_df)
        
        # 3. 清洗地理坐标
        cleaned_df = self.clean_coordinates(cleaned_df)
        
        # 4. 处理缺失值
        cleaned_df = self.handle_missing_values(cleaned_df)
        
        # 5. 移除重复数据
        cleaned_df = self.remove_duplicates(cleaned_df)
        
        # 6. 移除异常值
        cleaned_df = self.remove_outliers(cleaned_df)
        
        # 7. 数据验证
        validation_results = self.validate_cleaned_data(cleaned_df)
        self.log_info(f"Data cleaning completed. Validation results: {validation_results}")
        
        return cleaned_df
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名
        
        Args:
            df: DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        self.log_info("Standardizing column names")
        
        # 清理列名
        df.columns = [col.strip() for col in df.columns]
        
        # 应用列名映射
        for old_name, new_name in COLUMN_MAPPING.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                self.log_info(f"Renamed column: {old_name} -> {new_name}")
        
        return df
    
    @log_execution_time
    def process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理时间字段
        
        Args:
            df: DataFrame
            
        Returns:
            处理后的DataFrame
        """
        self.log_info("Processing datetime fields")
        
        # 确保时间列存在
        time_col = 'datetime' if 'datetime' in df.columns else 'Date/Time'
        
        if time_col not in df.columns:
            self.log_warning(f"Time column {time_col} not found")
            return df
        
        try:
            # 转换为datetime - 优化版本，跳过时区处理
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # 提取时间特征 - 批量处理
            df['hour'] = df[time_col].dt.hour
            df['day'] = df[time_col].dt.day
            df['weekday'] = df[time_col].dt.weekday
            df['month'] = df[time_col].dt.month
            df['quarter'] = df[time_col].dt.quarter
            df['year'] = df[time_col].dt.year
            df['date'] = df[time_col].dt.date
            
            # 添加业务特征 - 使用向量化操作
            df['is_weekend'] = (df['weekday'] >= 5).astype('int8')  # 使用int8节省内存
            df['is_peak_hour'] = self._is_peak_hour(df['hour']).astype('int8')
            
            self.log_info("Datetime processing completed successfully")
            
        except Exception as e:
            self.log_error(f"Error processing datetime: {str(e)}")
        
        return df
    
    def _is_peak_hour(self, hour_series: pd.Series) -> pd.Series:
        """
        判断是否为高峰时段
        
        Args:
            hour_series: 小时序列
            
        Returns:
            布尔序列
        """
        # 早高峰: 7-9点, 晚高峰: 17-19点
        return (hour_series.isin([7, 8, 9]) | hour_series.isin([17, 18, 19]))
    
    @log_execution_time
    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗地理坐标
        
        Args:
            df: DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        self.log_info("Cleaning geographic coordinates")
        
        lat_col = 'latitude' if 'latitude' in df.columns else 'Lat'
        lon_col = 'longitude' if 'longitude' in df.columns else 'Lon'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            self.log_warning("Latitude or longitude columns not found")
            return df
        
        initial_count = len(df)
        
        # 优化版本：一次性处理所有坐标验证
        # 创建综合mask，包含所有验证条件
        valid_mask = (
            # 非缺失值
            df[lat_col].notna() & df[lon_col].notna() &
            # NYC边界检查
            (df[lat_col] >= NYC_BOUNDS['min_lat']) & (df[lat_col] <= NYC_BOUNDS['max_lat']) &
            (df[lon_col] >= NYC_BOUNDS['min_lon']) & (df[lon_col] <= NYC_BOUNDS['max_lon']) &
            # 非零坐标
            ~((df[lat_col] == 0) & (df[lon_col] == 0))
        )
        
        # 一次性应用所有过滤条件
        df = df[valid_mask]
        
        # 移除重复坐标
        df = df.drop_duplicates(subset=[lat_col, lon_col])
        
        removed_count = initial_count - len(df)
        self.log_info(f"Removed {removed_count} invalid coordinate records")
        
        return df
    
    @log_execution_time
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: DataFrame
            
        Returns:
            处理后的DataFrame
        """
        self.log_info("Handling missing values")
        
        # 统计缺失值
        missing_counts = df.isnull().sum()
        self.log_info(f"Missing values before cleaning:\n{missing_counts[missing_counts > 0]}")
        
        # 优化版本：一次性处理所有缺失值
        # 收集所有需要检查的列
        cols_to_check = []
        
        # 时间列
        time_cols = ['datetime', 'Date/Time']
        for col in time_cols:
            if col in df.columns:
                cols_to_check.append(col)
        
        # 坐标列
        coord_cols = ['latitude', 'longitude', 'Lat', 'Lon']
        for col in coord_cols:
            if col in df.columns:
                cols_to_check.append(col)
        
        # 一次性移除所有缺失值
        if cols_to_check:
            df = df.dropna(subset=cols_to_check)
        
        # 对于Base列，用众数填充
        base_col = 'base_station' if 'base_station' in df.columns else 'Base'
        if base_col in df.columns and df[base_col].isnull().sum() > 0:
            mode_value = df[base_col].mode()[0]
            df[base_col] = df[base_col].fillna(mode_value)
            self.log_info(f"Filled missing values in {base_col} with mode: {mode_value}")
        
        # 对于数值列，用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                self.log_info(f"Filled missing values in {col} with median: {median_value}")
        
        remaining_missing = df.isnull().sum().sum()
        self.log_info(f"Remaining missing values after cleaning: {remaining_missing}")
        
        return df
    
    @log_execution_time
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        移除重复数据
        
        Args:
            df: DataFrame
            
        Returns:
            去重后的DataFrame
        """
        self.log_info("Removing duplicate records")
        
        initial_count = len(df)
        
        # 移除完全重复的行
        df = df.drop_duplicates()
        
        # 移除基于关键字段的重复
        key_columns = ['datetime', 'latitude', 'longitude']
        if all(col in df.columns for col in key_columns):
            df = df.drop_duplicates(subset=key_columns, keep='first')
        
        removed_count = initial_count - len(df)
        self.log_info(f"Removed {removed_count} duplicate records")
        
        return df
    
    @log_execution_time
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            df: DataFrame
            method: 异常值检测方法 ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            移除异常值后的DataFrame
        """
        self.log_info(f"Removing outliers using {method} method")
        
        initial_count = len(df)
        
        if method == 'iqr':
            df = self._remove_outliers_iqr(df)
        elif method == 'zscore':
            df = self._remove_outliers_zscore(df)
        elif method == 'isolation_forest':
            df = self._remove_outliers_isolation_forest(df)
        else:
            self.log_warning(f"Unknown outlier removal method: {method}")
        
        removed_count = initial_count - len(df)
        self.log_info(f"Removed {removed_count} outlier records")
        
        return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用IQR方法移除异常值
        
        Args:
            df: DataFrame
            
        Returns:
            处理后的DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 只对关键数值列进行异常值检测
        key_numeric_cols = ['latitude', 'longitude']
        key_numeric_cols = [col for col in key_numeric_cols if col in numeric_cols]
        
        for col in key_numeric_cols:
            if col in ['hour', 'day', 'weekday', 'month', 'quarter', 'year', 'is_weekend', 'is_peak_hour']:
                continue  # 跳过分类变量
            
            # 使用更宽松的阈值
            Q1 = df[col].quantile(0.05)  # 从0.25改为0.05
            Q3 = df[col].quantile(0.95)  # 从0.75改为0.95
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 2.0 * IQR  # 从1.5改为2.0
            upper_bound = Q3 + 2.0 * IQR  # 从1.5改为2.0
            
            # 记录移除的记录数
            before_count = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after_count = len(df)
            
            if before_count != after_count:
                self.log_info(f"Removed {before_count - after_count} outliers from {col}")
        
        return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        使用Z-score方法移除异常值
        
        Args:
            df: DataFrame
            threshold: Z-score阈值
            
        Returns:
            处理后的DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['hour', 'day', 'weekday', 'month', 'quarter', 'year', 'is_weekend', 'is_peak_hour']:
                continue
            
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < threshold]
        
        return df
    
    def _remove_outliers_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用Isolation Forest方法移除异常值
        
        Args:
            df: DataFrame
            
        Returns:
            处理后的DataFrame
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in 
                          ['hour', 'day', 'weekday', 'month', 'quarter', 'year', 'is_weekend', 'is_peak_hour']]
            
            if len(numeric_cols) == 0:
                return df
            
            # 准备数据
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # 训练Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(X)
            
            # 移除异常值
            df = df[outliers == 1]
            
        except ImportError:
            self.log_warning("IsolationForest not available, skipping outlier removal")
        except Exception as e:
            self.log_error(f"Error in isolation forest outlier removal: {str(e)}")
        
        return df
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证清洗后的数据
        
        Args:
            df: 清洗后的DataFrame
            
        Returns:
            验证结果字典
        """
        self.log_info("Validating cleaned data")
        
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'coordinate_bounds_check': True,
            'datetime_format_check': True,
            'issues': []
        }
        
        # 检查坐标范围
        lat_col = 'latitude' if 'latitude' in df.columns else 'Lat'
        lon_col = 'longitude' if 'longitude' in df.columns else 'Lon'
        
        if lat_col in df.columns and lon_col in df.columns:
            lat_in_bounds = (
                (df[lat_col] >= NYC_BOUNDS['min_lat']) & 
                (df[lat_col] <= NYC_BOUNDS['max_lat'])
            ).all()
            lon_in_bounds = (
                (df[lon_col] >= NYC_BOUNDS['min_lon']) & 
                (df[lon_col] <= NYC_BOUNDS['max_lon'])
            ).all()
            
            validation_results['coordinate_bounds_check'] = lat_in_bounds and lon_in_bounds
            
            if not validation_results['coordinate_bounds_check']:
                validation_results['issues'].append("Coordinates out of NYC bounds")
        
        # 检查时间格式
        time_col = 'datetime' if 'datetime' in df.columns else 'Date/Time'
        if time_col in df.columns:
            try:
                pd.to_datetime(df[time_col])
            except:
                validation_results['datetime_format_check'] = False
                validation_results['issues'].append("Invalid datetime format")
        
        # 检查数据完整性
        if validation_results['total_rows'] == 0:
            validation_results['issues'].append("No data remaining after cleaning")
        
        if validation_results['duplicate_rows'] > 0:
            validation_results['issues'].append(f"Found {validation_results['duplicate_rows']} duplicate rows")
        
        self.log_info(f"Validation completed. Found {len(validation_results['issues'])} issues")
        
        return validation_results
    
    @log_execution_time
    def create_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        创建数据清洗报告
        
        Args:
            original_df: 原始DataFrame
            cleaned_df: 清洗后的DataFrame
            
        Returns:
            清洗报告字典
        """
        self.log_info("Creating data cleaning report")
        
        report = {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_removed': original_df.shape[1] - cleaned_df.shape[1],
            'removal_percentage': {
                'rows': (original_df.shape[0] - cleaned_df.shape[0]) / original_df.shape[0] * 100,
                'columns': (original_df.shape[1] - cleaned_df.shape[1]) / original_df.shape[1] * 100
            },
            'missing_values_before': original_df.isnull().sum().to_dict(),
            'missing_values_after': cleaned_df.isnull().sum().to_dict(),
            'duplicates_before': original_df.duplicated().sum(),
            'duplicates_after': cleaned_df.duplicated().sum(),
            'data_quality_improvements': []
        }
        
        # 计算数据质量改进
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        
        if original_missing > cleaned_missing:
            report['data_quality_improvements'].append(
                f"Reduced missing values from {original_missing} to {cleaned_missing}"
            )
        
        if report['duplicates_before'] > report['duplicates_after']:
            report['data_quality_improvements'].append(
                f"Removed {report['duplicates_before'] - report['duplicates_after']} duplicate records"
            )
        
        self.log_info("Data cleaning report created successfully")
        return report

# 便捷函数
def clean_uber_data_sample(df: pd.DataFrame, sample_size: int = 50000) -> pd.DataFrame:
    """
    清洗Uber数据样本
    
    Args:
        df: 原始DataFrame
        sample_size: 样本大小
        
    Returns:
        清洗后的DataFrame
    """
    cleaner = DataCleaner()
    
    # 先采样再清洗
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return cleaner.clean_uber_data(df)

def get_cleaning_summary(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    获取数据清洗摘要
    
    Args:
        original_df: 原始DataFrame
        cleaned_df: 清洗后的DataFrame
        
    Returns:
        清洗摘要字典
    """
    cleaner = DataCleaner()
    return cleaner.create_cleaning_report(original_df, cleaned_df) 
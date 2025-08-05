"""
数据加载器模块
负责从各种数据源加载和预处理数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import glob
import os
from datetime import datetime
import warnings

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_FILES, 
    COLUMN_MAPPING, NYC_BOUNDS, SAMPLING_CONFIG
)
from config.logging_config import LoggerMixin, log_execution_time, log_data_info

warnings.filterwarnings('ignore')

class DataLoader(LoggerMixin):
    """
    数据加载器类
    支持多种数据源和格式的数据加载
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        super().__init__()
        self.data_dir = data_dir or RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.log_info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    @log_execution_time
    def load_uber_data(self, file_pattern: str = "uber-raw-data-*.csv") -> pd.DataFrame:
        """
        加载Uber数据文件
        
        Args:
            file_pattern: 文件匹配模式
            
        Returns:
            合并后的DataFrame
        """
        self.log_info(f"Loading Uber data with pattern: {file_pattern}")
        
        # 查找匹配的文件
        file_paths = glob.glob(str(self.data_dir / file_pattern))
        
        if not file_paths:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        self.log_info(f"Found {len(file_paths)} files: {file_paths}")
        
        # 加载所有文件 - 优化版本
        dfs = []
        
        # 使用更高效的数据类型
        dtype_dict = {
            'Date/Time': 'object',
            'Lat': 'float32',  # 使用float32而不是float64
            'Lon': 'float32',
            'Base': 'category'  # 使用category类型节省内存
        }
        
        for file_path in file_paths:
            try:
                # 只读取必要的列，使用更高效的数据类型
                df = pd.read_csv(file_path, dtype=dtype_dict, usecols=['Date/Time', 'Lat', 'Lon', 'Base'])
                df['source_file'] = os.path.basename(file_path)
                dfs.append(df)
                self.log_info(f"Loaded {len(df)} records from {file_path}")
            except Exception as e:
                self.log_error(f"Error loading {file_path}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No data files could be loaded")
        
        # 合并所有数据
        combined_df = pd.concat(dfs, ignore_index=True)
        self.log_info(f"Combined data shape: {combined_df.shape}")
        
        # 标准化列名
        combined_df.columns = [col.strip() for col in combined_df.columns]
        
        return combined_df
    
    @log_execution_time
    def load_single_file(self, filename: str) -> pd.DataFrame:
        """
        加载单个数据文件
        
        Args:
            filename: 文件名
            
        Returns:
            DataFrame
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.log_info(f"Loading single file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = filename
            self.log_info(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            self.log_error(f"Error loading {filename}: {str(e)}")
            raise
    
    @log_execution_time
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        加载处理后的数据
        
        Args:
            filename: 文件名或glob模式
            
        Returns:
            DataFrame
        """
        # 检查是否是glob模式
        if '*' in filename:
            # 使用glob模式查找文件
            file_paths = glob.glob(str(self.processed_dir / filename))
            if not file_paths:
                raise FileNotFoundError(f"No processed files found matching pattern: {filename}")
            
            # 选择最新的文件
            file_paths.sort()
            file_path = Path(file_paths[-1])
            self.log_info(f"Found {len(file_paths)} files, using latest: {file_path}")
        else:
            # 使用精确文件名
            file_path = self.processed_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Processed file not found: {file_path}")
        
        self.log_info(f"Loading processed data: {file_path}")
        
        try:
            if str(file_path).endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif str(file_path).endswith('.pkl'):
                df = pd.read_pickle(file_path)
            else:
                df = pd.read_csv(file_path)
            
            self.log_info(f"Loaded processed data shape: {df.shape}")
            return df
        except Exception as e:
            self.log_error(f"Error loading processed data {file_path}: {str(e)}")
            raise
    
    @log_execution_time
    def get_latest_processed_data(self, pattern: str = "featured_uber_data_*.parquet") -> pd.DataFrame:
        """
        获取最新的处理后数据
        
        Args:
            pattern: 文件模式
            
        Returns:
            DataFrame
        """
        return self.load_processed_data(pattern)
    
    @log_execution_time
    def save_processed_data(self, df: pd.DataFrame, filename: str, 
                          format: str = 'parquet') -> None:
        """
        保存处理后的数据
        
        Args:
            df: 要保存的DataFrame
            filename: 文件名
            format: 保存格式 ('csv', 'parquet', 'pkl')
        """
        file_path = self.processed_dir / filename
        
        self.log_info(f"Saving processed data to: {file_path}")
        
        try:
            if format == 'parquet':
                df.to_parquet(file_path, index=False)
            elif format == 'pkl':
                df.to_pickle(file_path)
            else:
                df.to_csv(file_path, index=False)
            
            self.log_info(f"Successfully saved data with shape {df.shape}")
        except Exception as e:
            self.log_error(f"Error saving data: {str(e)}")
            raise
    
    @log_execution_time
    def load_all_data_sources(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有数据源
        
        Returns:
            包含所有数据源的字典
        """
        self.log_info("Loading all data sources")
        
        data_sources = {}
        
        # 加载Uber数据
        try:
            uber_data = self.load_uber_data()
            data_sources['uber'] = uber_data
        except Exception as e:
            self.log_warning(f"Failed to load Uber data: {str(e)}")
        
        # 加载其他数据源
        other_patterns = [
            "other-*.csv",
            "FHV-*.csv"
        ]
        
        for pattern in other_patterns:
            try:
                files = glob.glob(str(self.data_dir / pattern))
                for file_path in files:
                    filename = os.path.basename(file_path)
                    df = pd.read_csv(file_path)
                    data_sources[filename] = df
                    self.log_info(f"Loaded {filename} with shape {df.shape}")
            except Exception as e:
                self.log_warning(f"Failed to load pattern {pattern}: {str(e)}")
        
        self.log_info(f"Loaded {len(data_sources)} data sources")
        return data_sources
    
    @log_execution_time
    def sample_data(self, df: pd.DataFrame, 
                   sample_size: Optional[int] = None,
                   random_state: Optional[int] = None) -> pd.DataFrame:
        """
        对数据进行采样
        
        Args:
            df: 原始DataFrame
            sample_size: 采样大小
            random_state: 随机种子
            
        Returns:
            采样后的DataFrame
        """
        if sample_size is None:
            sample_size = SAMPLING_CONFIG['sample_size']
        if random_state is None:
            random_state = SAMPLING_CONFIG['random_state']
        
        if len(df) <= sample_size:
            self.log_info(f"Data size ({len(df)}) <= sample size ({sample_size}), returning original data")
            return df
        
        self.log_info(f"Sampling {sample_size} records from {len(df)} total records")
        
        sampled_df = df.sample(n=sample_size, random_state=random_state)
        self.log_info(f"Sampling completed, new shape: {sampled_df.shape}")
        
        return sampled_df
    
    @log_execution_time
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        验证数据质量
        
        Args:
            df: 要验证的DataFrame
            
        Returns:
            验证结果字典
        """
        self.log_info("Validating data quality")
        
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'issues': []
        }
        
        # 检查必需列
        required_columns = ['Date/Time', 'Lat', 'Lon', 'Base']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # 检查地理坐标范围
        if 'Lat' in df.columns and 'Lon' in df.columns:
            lat_out_of_bounds = (
                (df['Lat'] < NYC_BOUNDS['min_lat']) | 
                (df['Lat'] > NYC_BOUNDS['max_lat'])
            ).sum()
            lon_out_of_bounds = (
                (df['Lon'] < NYC_BOUNDS['min_lon']) | 
                (df['Lon'] > NYC_BOUNDS['max_lon'])
            ).sum()
            
            if lat_out_of_bounds > 0:
                validation_results['issues'].append(f"Latitude out of bounds: {lat_out_of_bounds} records")
            if lon_out_of_bounds > 0:
                validation_results['issues'].append(f"Longitude out of bounds: {lon_out_of_bounds} records")
        
        # 检查时间格式
        if 'Date/Time' in df.columns:
            try:
                pd.to_datetime(df['Date/Time'])
            except Exception as e:
                validation_results['issues'].append(f"Invalid datetime format: {str(e)}")
        
        self.log_info(f"Validation completed. Found {len(validation_results['issues'])} issues")
        
        return validation_results
    
    @log_execution_time
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        获取数据摘要信息
        
        Args:
            df: DataFrame
            
        Returns:
            数据摘要字典
        """
        self.log_info("Generating data summary")
        
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # 数值列摘要
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # 分类列摘要
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                summary['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        self.log_info("Data summary generated successfully")
        return summary
    
    def list_available_files(self) -> List[str]:
        """
        列出可用的数据文件
        
        Returns:
            文件列表
        """
        files = []
        for pattern in ['*.csv', '*.parquet', '*.pkl']:
            files.extend(glob.glob(str(self.data_dir / pattern)))
        
        return [os.path.basename(f) for f in files]
    
    @log_execution_time
    def create_data_catalog(self) -> pd.DataFrame:
        """
        创建数据目录
        
        Returns:
            包含所有数据文件信息的DataFrame
        """
        self.log_info("Creating data catalog")
        
        catalog_data = []
        
        for file_path in self.data_dir.glob('*'):
            if file_path.is_file():
                try:
                    file_info = {
                        'filename': file_path.name,
                        'size_mb': file_path.stat().st_size / 1024**2,
                        'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime),
                        'file_type': file_path.suffix
                    }
                    
                    # 尝试读取文件信息
                    if file_path.suffix == '.csv':
                        try:
                            df = pd.read_csv(file_path, nrows=1)
                            file_info['columns'] = list(df.columns)
                            file_info['sample_rows'] = len(pd.read_csv(file_path))
                        except:
                            file_info['columns'] = []
                            file_info['sample_rows'] = 0
                    
                    catalog_data.append(file_info)
                    
                except Exception as e:
                    self.log_warning(f"Error processing {file_path}: {str(e)}")
        
        catalog_df = pd.DataFrame(catalog_data)
        self.log_info(f"Data catalog created with {len(catalog_df)} files")
        
        return catalog_df

# 便捷函数
def load_uber_data_sample(sample_size: int = 50000) -> pd.DataFrame:
    """
    加载Uber数据样本
    
    Args:
        sample_size: 样本大小
        
    Returns:
        采样后的DataFrame
    """
    loader = DataLoader()
    df = loader.load_uber_data()
    return loader.sample_data(df, sample_size=sample_size)

def load_and_validate_data() -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    加载并验证数据
    
    Returns:
        (DataFrame, 验证结果)
    """
    loader = DataLoader()
    df = loader.load_uber_data()
    validation_results = loader.validate_data(df)
    return df, validation_results 
"""
辅助函数模块

该模块提供各种实用工具函数，包括：
- 数据处理辅助函数
- 文件操作函数
- 时间处理函数
- 数学计算函数
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path, file_type=None):
    """
    通用数据加载函数
    
    Parameters:
    -----------
    file_path : str
        文件路径
    file_type : str, optional
        文件类型，如果为None则自动推断
        
    Returns:
    --------
    pandas.DataFrame : 加载的数据
    """
    if file_type is None:
        file_type = file_path.split('.')[-1].lower()
    
    try:
        if file_type in ['csv']:
            return pd.read_csv(file_path)
        elif file_type in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        elif file_type in ['json']:
            return pd.read_json(file_path)
        elif file_type in ['parquet']:
            return pd.read_parquet(file_path)
        elif file_type in ['pkl', 'pickle']:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None


def save_data(data, file_path, file_type=None):
    """
    通用数据保存函数
    
    Parameters:
    -----------
    data : pandas.DataFrame or object
        要保存的数据
    file_path : str
        文件路径
    file_type : str, optional
        文件类型，如果为None则自动推断
    """
    if file_type is None:
        file_type = file_path.split('.')[-1].lower()
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_type in ['csv']:
            data.to_csv(file_path, index=False)
        elif file_type in ['xlsx', 'xls']:
            data.to_excel(file_path, index=False)
        elif file_type in ['json']:
            data.to_json(file_path, orient='records')
        elif file_type in ['parquet']:
            data.to_parquet(file_path, index=False)
        elif file_type in ['pkl', 'pickle']:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
            
        print(f"数据已保存到: {file_path}")
    except Exception as e:
        print(f"保存文件 {file_path} 时出错: {e}")


def clean_dataframe(df, remove_duplicates=True, handle_missing='drop'):
    """
    数据框清洗函数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        要清洗的数据框
    remove_duplicates : bool
        是否移除重复行
    handle_missing : str
        缺失值处理方式：'drop', 'fill_mean', 'fill_median', 'fill_mode'
        
    Returns:
    --------
    pandas.DataFrame : 清洗后的数据框
    """
    df_clean = df.copy()
    
    # 移除重复行
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            print(f"移除了 {removed_rows} 行重复数据")
    
    # 处理缺失值
    if handle_missing == 'drop':
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            print(f"移除了 {removed_rows} 行包含缺失值的数据")
    elif handle_missing == 'fill_mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                print(f"用均值填充了列 {col} 的缺失值")
    elif handle_missing == 'fill_median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"用中位数填充了列 {col} 的缺失值")
    elif handle_missing == 'fill_mode':
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else None
                if mode_value is not None:
                    df_clean[col].fillna(mode_value, inplace=True)
                    print(f"用众数填充了列 {col} 的缺失值")
    
    return df_clean


def extract_time_features(df, time_column):
    """
    从时间列提取时间特征
    
    Parameters:
    -----------
    df : pandas.DataFrame
        数据框
    time_column : str
        时间列名
        
    Returns:
    --------
    pandas.DataFrame : 包含时间特征的数据框
    """
    df_features = df.copy()
    
    # 确保时间列格式正确
    if not pd.api.types.is_datetime64_any_dtype(df_features[time_column]):
        df_features[time_column] = pd.to_datetime(df_features[time_column])
    
    # 提取时间特征
    df_features[f'{time_column}_year'] = df_features[time_column].dt.year
    df_features[f'{time_column}_month'] = df_features[time_column].dt.month
    df_features[f'{time_column}_day'] = df_features[time_column].dt.day
    df_features[f'{time_column}_hour'] = df_features[time_column].dt.hour
    df_features[f'{time_column}_minute'] = df_features[time_column].dt.minute
    df_features[f'{time_column}_weekday'] = df_features[time_column].dt.dayofweek
    df_features[f'{time_column}_week'] = df_features[time_column].dt.isocalendar().week
    df_features[f'{time_column}_quarter'] = df_features[time_column].dt.quarter
    
    # 是否为周末
    df_features[f'{time_column}_is_weekend'] = df_features[time_column].dt.weekday >= 5
    
    # 是否为工作日
    df_features[f'{time_column}_is_workday'] = df_features[time_column].dt.weekday < 5
    
    # 时间段分类
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df_features[f'{time_column}_time_period'] = df_features[time_column].dt.hour.apply(get_time_period)
    
    return df_features


def calculate_distance(lat1, lng1, lat2, lng2, method='haversine'):
    """
    计算两点间距离
    
    Parameters:
    -----------
    lat1, lng1 : float
        第一个点的经纬度
    lat2, lng2 : float
        第二个点的经纬度
    method : str
        计算方法：'haversine', 'euclidean'
        
    Returns:
    --------
    float : 距离（公里）
    """
    if method == 'haversine':
        # 使用Haversine公式计算球面距离
        R = 6371  # 地球半径（公里）
        
        lat1_rad = np.radians(lat1)
        lng1_rad = np.radians(lng1)
        lat2_rad = np.radians(lat2)
        lng2_rad = np.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance = R * c
        return distance
    
    elif method == 'euclidean':
        # 使用欧几里得距离（近似）
        distance = np.sqrt((lat2 - lat1)**2 + (lng2 - lng1)**2) * 111  # 1度约等于111公里
        return distance
    
    else:
        raise ValueError(f"不支持的距离计算方法: {method}")


def detect_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    检测异常值
    
    Parameters:
    -----------
    df : pandas.DataFrame
        数据框
    columns : list, optional
        要检测的列名列表
    method : str
        检测方法：'iqr', 'zscore'
    threshold : float
        阈值
        
    Returns:
    --------
    dict : 异常值信息
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers_info = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > threshold]
        
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        outliers_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'indices': outliers.index.tolist()
        }
    
    return outliers_info


def sample_data(df, method='random', size=1000, **kwargs):
    """
    数据采样函数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        数据框
    method : str
        采样方法：'random', 'systematic', 'stratified'
    size : int
        采样大小
    **kwargs : dict
        其他参数
        
    Returns:
    --------
    pandas.DataFrame : 采样后的数据框
    """
    if method == 'random':
        return df.sample(n=min(size, len(df)), random_state=kwargs.get('random_state', 42))
    
    elif method == 'systematic':
        step = len(df) // size
        indices = range(0, len(df), step)[:size]
        return df.iloc[indices]
    
    elif method == 'stratified':
        stratify_col = kwargs.get('stratify_column')
        if stratify_col is None:
            raise ValueError("分层采样需要指定 stratify_column 参数")
        
        return df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), size // len(df[stratify_col].unique())))
        )
    
    else:
        raise ValueError(f"不支持的采样方法: {method}")


def format_number(number, decimals=2):
    """
    格式化数字显示
    
    Parameters:
    -----------
    number : float
        要格式化的数字
    decimals : int
        小数位数
        
    Returns:
    --------
    str : 格式化后的字符串
    """
    if abs(number) >= 1e6:
        return f"{number/1e6:.{decimals}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{decimals}f}K"
    else:
        return f"{number:.{decimals}f}"


def create_summary_statistics(df, columns=None):
    """
    创建数据摘要统计
    
    Parameters:
    -----------
    df : pandas.DataFrame
        数据框
    columns : list, optional
        要统计的列名列表
        
    Returns:
    --------
    dict : 摘要统计信息
    """
    if columns is None:
        columns = df.columns.tolist()
    
    summary = {}
    
    for col in columns:
        col_data = df[col]
        
        if pd.api.types.is_numeric_dtype(col_data):
            summary[col] = {
                'type': 'numeric',
                'count': len(col_data),
                'missing': col_data.isnull().sum(),
                'missing_pct': col_data.isnull().sum() / len(col_data) * 100,
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75)
            }
        else:
            summary[col] = {
                'type': 'categorical',
                'count': len(col_data),
                'missing': col_data.isnull().sum(),
                'missing_pct': col_data.isnull().sum() / len(col_data) * 100,
                'unique_values': col_data.nunique(),
                'most_common': col_data.mode()[0] if len(col_data.mode()) > 0 else None,
                'most_common_count': col_data.value_counts().iloc[0] if len(col_data.value_counts()) > 0 else 0
            }
    
    return summary


def print_summary_statistics(summary):
    """
    打印摘要统计信息
    
    Parameters:
    -----------
    summary : dict
        摘要统计信息
    """
    print("数据摘要统计")
    print("=" * 60)
    
    for col, stats in summary.items():
        print(f"\n列名: {col}")
        print(f"类型: {stats['type']}")
        print(f"总数量: {stats['count']}")
        print(f"缺失值: {stats['missing']} ({stats['missing_pct']:.2f}%)")
        
        if stats['type'] == 'numeric':
            print(f"均值: {stats['mean']:.4f}")
            print(f"中位数: {stats['median']:.4f}")
            print(f"标准差: {stats['std']:.4f}")
            print(f"最小值: {stats['min']:.4f}")
            print(f"最大值: {stats['max']:.4f}")
            print(f"25分位数: {stats['q25']:.4f}")
            print(f"75分位数: {stats['q75']:.4f}")
        else:
            print(f"唯一值数量: {stats['unique_values']}")
            print(f"最常见值: {stats['most_common']}")
            print(f"最常见值频次: {stats['most_common_count']}")


def validate_data_types(df, expected_types):
    """
    验证数据类型
    
    Parameters:
    -----------
    df : pandas.DataFrame
        数据框
    expected_types : dict
        期望的数据类型字典
        
    Returns:
    --------
    dict : 验证结果
    """
    validation_results = {}
    
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            validation_results[col] = {
                'status': 'missing',
                'message': f'列 {col} 不存在'
            }
        else:
            actual_type = str(df[col].dtype)
            if actual_type == expected_type:
                validation_results[col] = {
                    'status': 'valid',
                    'message': f'类型匹配: {actual_type}'
                }
            else:
                validation_results[col] = {
                    'status': 'invalid',
                    'message': f'类型不匹配: 期望 {expected_type}, 实际 {actual_type}'
                }
    
    return validation_results


if __name__ == "__main__":
    # 示例用法
    print("辅助函数模块")
    print("请导入并使用各种辅助函数") 
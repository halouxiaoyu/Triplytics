"""
探索性数据分析 (EDA) 模块
提供全面的数据探索和分析功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.logging_config import LoggerMixin, log_execution_time
from config.settings import VISUALIZATION_CONFIG

warnings.filterwarnings('ignore')

class ExploratoryDataAnalysis(LoggerMixin):
    """
    探索性数据分析类
    提供全面的数据探索功能
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化EDA分析器
        
        Args:
            df: 要分析的DataFrame
        """
        super().__init__()
        self.df = df.copy()
        self.log_info(f"EDA initialized with data shape: {df.shape}")
    
    @log_execution_time
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        生成数据摘要报告
        
        Returns:
            包含各种统计信息的字典
        """
        self.log_info("Generating comprehensive summary report")
        
        report = {
            'basic_info': self._get_basic_info(),
            'missing_data': self._analyze_missing_data(),
            'data_types': self._analyze_data_types(),
            'numerical_analysis': self._analyze_numerical_features(),
            'categorical_analysis': self._analyze_categorical_features(),
            'temporal_analysis': self._analyze_temporal_features(),
            'spatial_analysis': self._analyze_spatial_features(),
            'correlation_analysis': self._analyze_correlations(),
            'outlier_analysis': self._analyze_outliers(),
            'data_quality': self._assess_data_quality()
        }
        
        self.log_info("Summary report generated successfully")
        return report
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """获取基本信息"""
        return {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': self.df.duplicated().sum(),
            'unique_values_per_column': self.df.nunique().to_dict()
        }
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """分析缺失数据"""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist(),
            'total_missing': missing_counts.sum()
        }
    
    def _analyze_data_types(self) -> Dict[str, Any]:
        """分析数据类型"""
        dtypes = self.df.dtypes
        type_counts = dtypes.value_counts()
        
        return {
            'dtypes': dtypes.to_dict(),
            'type_counts': type_counts.to_dict(),
            'numerical_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': self.df.select_dtypes(include=['datetime64']).columns.tolist()
        }
    
    def _analyze_numerical_features(self) -> Dict[str, Any]:
        """分析数值特征"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {'message': 'No numerical columns found'}
        
        analysis = {}
        
        # 描述性统计
        analysis['descriptive_stats'] = self.df[numerical_cols].describe().to_dict()
        
        # 分布分析
        for col in numerical_cols:
            analysis[f'{col}_distribution'] = {
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'iqr': self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
            }
        
        return analysis
    
    def _analyze_categorical_features(self) -> Dict[str, Any]:
        """分析分类特征"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return {'message': 'No categorical columns found'}
        
        analysis = {}
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            analysis[col] = {
                'unique_count': self.df[col].nunique(),
                'top_values': value_counts.head(10).to_dict(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'least_common': value_counts.index[-1] if len(value_counts) > 0 else None
            }
        
        return analysis
    
    def _analyze_temporal_features(self) -> Dict[str, Any]:
        """分析时间特征"""
        analysis = {}
        
        # 检查是否有时间列
        time_cols = ['datetime', 'Date/Time']
        time_col = None
        for col in time_cols:
            if col in self.df.columns:
                time_col = col
                break
        
        if time_col is None:
            return {'message': 'No temporal columns found'}
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.df[time_col]):
            self.df[time_col] = pd.to_datetime(self.df[time_col], errors='coerce')
        
        analysis['time_range'] = {
            'start': self.df[time_col].min(),
            'end': self.df[time_col].max(),
            'duration_days': (self.df[time_col].max() - self.df[time_col].min()).days
        }
        
        # 时间模式分析
        if 'hour' in self.df.columns:
            hourly_counts = self.df['hour'].value_counts().sort_index()
            analysis['hourly_pattern'] = {
                'peak_hours': hourly_counts.nlargest(3).index.tolist(),
                'off_peak_hours': hourly_counts.nsmallest(3).index.tolist(),
                'hourly_distribution': hourly_counts.to_dict()
            }
        
        if 'weekday' in self.df.columns:
            weekday_counts = self.df['weekday'].value_counts().sort_index()
            analysis['weekday_pattern'] = {
                'busiest_day': weekday_counts.idxmax(),
                'quietest_day': weekday_counts.idxmin(),
                'weekday_distribution': weekday_counts.to_dict()
            }
        
        return analysis
    
    def _analyze_spatial_features(self) -> Dict[str, Any]:
        """分析空间特征"""
        analysis = {}
        
        # 检查地理坐标列
        lat_cols = ['latitude', 'Lat']
        lon_cols = ['longitude', 'Lon']
        
        lat_col = None
        lon_col = None
        
        for col in lat_cols:
            if col in self.df.columns:
                lat_col = col
                break
        
        for col in lon_cols:
            if col in self.df.columns:
                lon_col = col
                break
        
        if lat_col is None or lon_col is None:
            return {'message': 'No spatial columns found'}
        
        # 地理范围分析
        analysis['geographic_bounds'] = {
            'lat_range': (self.df[lat_col].min(), self.df[lat_col].max()),
            'lon_range': (self.df[lon_col].min(), self.df[lon_col].max()),
            'lat_center': self.df[lat_col].mean(),
            'lon_center': self.df[lon_col].mean()
        }
        
        # 聚类分析
        if 'cluster_8' in self.df.columns:
            cluster_counts = self.df['cluster_8'].value_counts()
            analysis['clustering'] = {
                'cluster_distribution': cluster_counts.to_dict(),
                'most_popular_cluster': cluster_counts.idxmax(),
                'least_popular_cluster': cluster_counts.idxmin(),
                'cluster_diversity': cluster_counts.nunique()
            }
        
        return analysis
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """分析相关性"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {'message': 'Insufficient numerical columns for correlation analysis'}
        
        # 计算相关性矩阵
        corr_matrix = self.df[numerical_cols].corr()
        
        # 找出强相关特征
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # 强相关阈值
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        }
    
    def _analyze_outliers(self) -> Dict[str, Any]:
        """分析异常值"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {'message': 'No numerical columns for outlier analysis'}
        
        outlier_analysis = {}
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            outlier_analysis[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_value': self.df[col].min(),
                'max_value': self.df[col].max()
            }
        
        return outlier_analysis
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """评估数据质量"""
        quality_score = 100
        
        # 检查缺失值
        missing_percentage = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_percentage > 10:
            quality_score -= 20
        elif missing_percentage > 5:
            quality_score -= 10
        
        # 检查重复值
        duplicate_percentage = (self.df.duplicated().sum() / len(self.df)) * 100
        if duplicate_percentage > 5:
            quality_score -= 15
        elif duplicate_percentage > 1:
            quality_score -= 5
        
        # 检查数据类型一致性
        inconsistent_types = 0
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # 检查是否可以转换为数值
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                    inconsistent_types += 1
                except:
                    pass
        
        if inconsistent_types > 0:
            quality_score -= 10
        
        return {
            'overall_quality_score': max(0, quality_score),
            'missing_data_impact': missing_percentage,
            'duplicate_data_impact': duplicate_percentage,
            'data_type_consistency': len(self.df.columns) - inconsistent_types,
            'recommendations': self._generate_quality_recommendations(quality_score)
        }
    
    def _generate_quality_recommendations(self, quality_score: float) -> List[str]:
        """生成数据质量建议"""
        recommendations = []
        
        if quality_score < 80:
            recommendations.append("数据质量较低，建议进行详细的数据清洗")
        
        if self.df.isnull().sum().sum() > 0:
            recommendations.append("存在缺失值，建议进行缺失值处理")
        
        if self.df.duplicated().sum() > 0:
            recommendations.append("存在重复数据，建议进行去重处理")
        
        if quality_score >= 90:
            recommendations.append("数据质量良好，可以直接用于建模")
        
        return recommendations
    
    @log_execution_time
    def create_visualizations(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """
        创建可视化图表
        
        Args:
            save_path: 保存路径
            
        Returns:
            图表文件路径字典
        """
        self.log_info("Creating comprehensive visualizations")
        
        # 设置绘图样式
        plt.style.use(VISUALIZATION_CONFIG['style'])
        
        saved_files = {}
        
        # 1. 数据概览图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 缺失值热力图
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            missing_data.plot(kind='bar', ax=axes[0, 0], color='red', alpha=0.7)
            axes[0, 0].set_title('Missing Values by Column')
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Missing Values by Column')
        
        # 数据类型分布
        dtypes = self.df.dtypes.value_counts()
        dtypes.plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
        axes[0, 1].set_title('Data Types Distribution')
        
        # 数值特征分布
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            self.df[numerical_cols].boxplot(ax=axes[1, 0])
            axes[1, 0].set_title('Numerical Features Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Numerical Features', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Numerical Features Distribution')
        
        # 相关性热力图
        if len(numerical_cols) > 1:
            corr_matrix = self.df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Correlation Matrix')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient Numerical Features', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            overview_path = f"{save_path}/data_overview.png"
            plt.savefig(overview_path, dpi=300, bbox_inches='tight')
            saved_files['overview'] = overview_path
        
        plt.show()
        
        # 2. 时间模式分析
        if 'hour' in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 每小时分布
            hourly_counts = self.df['hour'].value_counts().sort_index()
            hourly_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_title('Hourly Distribution')
            axes[0, 0].set_xlabel('Hour')
            axes[0, 0].set_ylabel('Count')
            
            # 星期分布
            if 'weekday' in self.df.columns:
                weekday_counts = self.df['weekday'].value_counts().sort_index()
                weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                weekday_counts.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
                axes[0, 1].set_title('Weekday Distribution')
                axes[0, 1].set_xlabel('Weekday')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_xticklabels(weekday_names)
            
            # 时间-星期热力图
            if 'weekday' in self.df.columns:
                hourly_weekday = self.df.groupby(['hour', 'weekday']).size().unstack(fill_value=0)
                sns.heatmap(hourly_weekday, ax=axes[1, 0], cmap='YlOrRd')
                axes[1, 0].set_title('Hour-Weekday Heatmap')
            
            # 月份分布
            if 'month' in self.df.columns:
                monthly_counts = self.df['month'].value_counts().sort_index()
                monthly_counts.plot(kind='bar', ax=axes[1, 1], color='lightgreen')
                axes[1, 1].set_title('Monthly Distribution')
                axes[1, 1].set_xlabel('Month')
                axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            
            if save_path:
                temporal_path = f"{save_path}/temporal_analysis.png"
                plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
                saved_files['temporal'] = temporal_path
            
            plt.show()
        
        # 3. 空间分析
        lat_col = 'latitude' if 'latitude' in self.df.columns else 'Lat'
        lon_col = 'longitude' if 'longitude' in self.df.columns else 'Lon'
        
        if lat_col in self.df.columns and lon_col in self.df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 散点图
            sample_size = min(10000, len(self.df))
            sample_df = self.df.sample(n=sample_size, random_state=42)
            
            scatter = axes[0].scatter(sample_df[lon_col], sample_df[lat_col], 
                                    c=sample_df['Base'].astype('category').cat.codes if 'Base' in sample_df.columns else 'blue',
                                    s=1, alpha=0.6, cmap='tab10')
            axes[0].set_title('Geographic Distribution')
            axes[0].set_xlabel('Longitude')
            axes[0].set_ylabel('Latitude')
            
            # 热力图
            axes[1].hist2d(sample_df[lon_col], sample_df[lat_col], bins=50, cmap='Reds')
            axes[1].set_title('Density Heatmap')
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            
            plt.tight_layout()
            
            if save_path:
                spatial_path = f"{save_path}/spatial_analysis.png"
                plt.savefig(spatial_path, dpi=300, bbox_inches='tight')
                saved_files['spatial'] = spatial_path
            
            plt.show()
        
        self.log_info(f"Visualizations created successfully. Saved {len(saved_files)} files.")
        return saved_files
    
    def print_summary(self):
        """打印分析摘要"""
        report = self.generate_summary_report()
        
        print("=" * 80)
        print("探索性数据分析 (EDA) 摘要报告")
        print("=" * 80)
        
        # 基本信息
        print(f"\n📊 基本信息:")
        print(f"  数据形状: {report['basic_info']['shape']}")
        print(f"  内存使用: {report['basic_info']['memory_usage_mb']:.2f} MB")
        print(f"  重复行数: {report['basic_info']['duplicate_rows']}")
        
        # 缺失值信息
        missing_info = report['missing_data']
        print(f"\n❌ 缺失值分析:")
        print(f"  总缺失值: {missing_info['total_missing']}")
        print(f"  有缺失值的列数: {len(missing_info['columns_with_missing'])}")
        
        # 数据质量
        quality_info = report['data_quality']
        print(f"\n🎯 数据质量评估:")
        print(f"  总体质量评分: {quality_info['overall_quality_score']}/100")
        print(f"  建议:")
        for rec in quality_info['recommendations']:
            print(f"    - {rec}")
        
        # 时间分析
        if 'time_range' in report['temporal_analysis']:
            temporal_info = report['temporal_analysis']
            print(f"\n⏰ 时间分析:")
            print(f"  时间跨度: {temporal_info['time_range']['duration_days']} 天")
            if 'hourly_pattern' in temporal_info:
                print(f"  高峰时段: {temporal_info['hourly_pattern']['peak_hours']}")
        
        # 空间分析
        if 'geographic_bounds' in report['spatial_analysis']:
            spatial_info = report['spatial_analysis']
            print(f"\n📍 空间分析:")
            bounds = spatial_info['geographic_bounds']
            print(f"  纬度范围: {bounds['lat_range']}")
            print(f"  经度范围: {bounds['lon_range']}")
        
        print("\n" + "=" * 80)

# 便捷函数
def quick_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    快速EDA分析
    
    Args:
        df: DataFrame
        
    Returns:
        EDA报告
    """
    eda = ExploratoryDataAnalysis(df)
    return eda.generate_summary_report()

def create_eda_visualizations(df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, str]:
    """
    创建EDA可视化
    
    Args:
        df: DataFrame
        save_path: 保存路径
        
    Returns:
        图表文件路径
    """
    eda = ExploratoryDataAnalysis(df)
    return eda.create_visualizations(save_path) 
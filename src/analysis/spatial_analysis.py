"""
空间分析模块
提供地理空间数据的分析和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap

from config.logging_config import LoggerMixin, log_execution_time
from config.settings import NYC_BOUNDS, CLUSTERING_CONFIG

warnings.filterwarnings('ignore')

class SpatialAnalysis(LoggerMixin):
    """
    空间分析类
    提供地理空间数据的分析功能
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化空间分析器
        
        Args:
            df: 包含地理坐标的DataFrame
        """
        super().__init__()
        self.df = df.copy()
        self.log_info(f"SpatialAnalysis initialized with data shape: {df.shape}")
        
        # 识别坐标列
        self.lat_col = self._find_lat_column()
        self.lon_col = self._find_lon_column()
        
        if self.lat_col is None or self.lon_col is None:
            self.log_warning("Latitude or longitude columns not found")
    
    def _find_lat_column(self) -> Optional[str]:
        """查找纬度列"""
        lat_candidates = ['latitude', 'Lat', 'lat']
        for col in lat_candidates:
            if col in self.df.columns:
                return col
        return None
    
    def _find_lon_column(self) -> Optional[str]:
        """查找经度列"""
        lon_candidates = ['longitude', 'Lon', 'lon']
        for col in lon_candidates:
            if col in self.df.columns:
                return col
        return None
    
    @log_execution_time
    def analyze_geographic_distribution(self) -> Dict[str, Any]:
        """
        分析地理分布
        
        Returns:
            地理分布分析结果
        """
        self.log_info("Analyzing geographic distribution")
        
        if self.lat_col is None or self.lon_col is None:
            return {'error': 'Latitude or longitude columns not found'}
        
        analysis = {
            'bounds': self._calculate_bounds(),
            'center': self._calculate_center(),
            'density': self._calculate_density(),
            'spread': self._calculate_spread(),
            'outliers': self._detect_geographic_outliers()
        }
        
        self.log_info("Geographic distribution analysis completed")
        return analysis
    
    def _calculate_bounds(self) -> Dict[str, Tuple[float, float]]:
        """计算地理边界"""
        return {
            'latitude': (self.df[self.lat_col].min(), self.df[self.lat_col].max()),
            'longitude': (self.df[self.lon_col].min(), self.df[self.lon_col].max()),
            'nyc_bounds': NYC_BOUNDS
        }
    
    def _calculate_center(self) -> Dict[str, float]:
        """计算地理中心"""
        return {
            'latitude': self.df[self.lat_col].mean(),
            'longitude': self.df[self.lon_col].mean(),
            'median_lat': self.df[self.lat_col].median(),
            'median_lon': self.df[self.lon_col].median()
        }
    
    def _calculate_density(self) -> Dict[str, Any]:
        """计算密度统计"""
        # 计算每个点周围的点数
        coords = self.df[[self.lat_col, self.lon_col]].values
        
        # 为了避免计算量过大，采样计算
        sample_size = min(1000, len(coords))
        sample_coords = coords[:sample_size]
        
        # 计算距离矩阵
        distances = cdist(sample_coords, coords)
        
        # 计算不同半径内的点数
        radius_1km = 0.01  # 约1公里
        radius_5km = 0.05  # 约5公里
        
        density_1km = (distances < radius_1km).sum(axis=1) - 1  # 减去自己
        density_5km = (distances < radius_5km).sum(axis=1) - 1
        
        return {
            'density_1km_mean': density_1km.mean(),
            'density_1km_std': density_1km.std(),
            'density_1km_max': density_1km.max(),
            'density_5km_mean': density_5km.mean(),
            'density_5km_std': density_5km.std(),
            'density_5km_max': density_5km.max()
        }
    
    def _calculate_spread(self) -> Dict[str, float]:
        """计算地理分布范围"""
        return {
            'lat_range': self.df[self.lat_col].max() - self.df[self.lat_col].min(),
            'lon_range': self.df[self.lon_col].max() - self.df[self.lon_col].min(),
            'lat_std': self.df[self.lat_col].std(),
            'lon_std': self.df[self.lon_col].std()
        }
    
    def _detect_geographic_outliers(self) -> Dict[str, Any]:
        """检测地理异常值"""
        # 使用IQR方法检测异常值
        lat_q1 = self.df[self.lat_col].quantile(0.25)
        lat_q3 = self.df[self.lat_col].quantile(0.75)
        lat_iqr = lat_q3 - lat_q1
        
        lon_q1 = self.df[self.lon_col].quantile(0.25)
        lon_q3 = self.df[self.lon_col].quantile(0.75)
        lon_iqr = lon_q3 - lon_q1
        
        lat_outliers = self.df[
            (self.df[self.lat_col] < lat_q1 - 1.5 * lat_iqr) |
            (self.df[self.lat_col] > lat_q3 + 1.5 * lat_iqr)
        ]
        
        lon_outliers = self.df[
            (self.df[self.lon_col] < lon_q1 - 1.5 * lon_iqr) |
            (self.df[self.lon_col] > lon_q3 + 1.5 * lon_iqr)
        ]
        
        return {
            'lat_outliers_count': len(lat_outliers),
            'lon_outliers_count': len(lon_outliers),
            'total_outliers': len(pd.concat([lat_outliers, lon_outliers]).drop_duplicates())
        }
    
    @log_execution_time
    def perform_clustering_analysis(self, method: str = 'kmeans', **kwargs) -> Dict[str, Any]:
        """
        执行聚类分析
        
        Args:
            method: 聚类方法 ('kmeans', 'dbscan')
            **kwargs: 聚类参数
            
        Returns:
            聚类分析结果
        """
        self.log_info(f"Performing {method} clustering analysis")
        
        if self.lat_col is None or self.lon_col is None:
            return {'error': 'Latitude or longitude columns not found'}
        
        coords = self.df[[self.lat_col, self.lon_col]].values
        
        if method == 'kmeans':
            return self._kmeans_clustering(coords, **kwargs)
        elif method == 'dbscan':
            return self._dbscan_clustering(coords, **kwargs)
        else:
            self.log_error(f"Unknown clustering method: {method}")
            return {'error': f'Unknown clustering method: {method}'}
    
    def _kmeans_clustering(self, coords: np.ndarray, n_clusters: int = 8) -> Dict[str, Any]:
        """K-means聚类"""
        kmeans_config = CLUSTERING_CONFIG['kmeans'].copy()
        kmeans_config['n_clusters'] = n_clusters
        
        kmeans = KMeans(**kmeans_config)
        cluster_labels = kmeans.fit_predict(coords)
        
        # 添加聚类标签到数据
        self.df[f'cluster_{n_clusters}'] = cluster_labels
        
        # 分析聚类结果
        cluster_counts = pd.Series(cluster_labels).value_counts()
        cluster_centers = kmeans.cluster_centers_
        
        # 计算每个聚类的统计信息
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = self.df[cluster_labels == i]
            cluster_stats[i] = {
                'count': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.df)) * 100,
                'center_lat': cluster_centers[i, 0],
                'center_lon': cluster_centers[i, 1],
                'radius': self._calculate_cluster_radius(cluster_data, cluster_centers[i])
            }
        
        return {
            'method': 'kmeans',
            'n_clusters': n_clusters,
            'cluster_counts': cluster_counts.to_dict(),
            'cluster_centers': cluster_centers.tolist(),
            'cluster_stats': cluster_stats,
            'inertia': kmeans.inertia_,
            'model': kmeans
        }
    
    def _dbscan_clustering(self, coords: np.ndarray, eps: float = 0.01, min_samples: int = 50) -> Dict[str, Any]:
        """DBSCAN聚类"""
        dbscan_config = CLUSTERING_CONFIG['dbscan'].copy()
        dbscan_config.update({'eps': eps, 'min_samples': min_samples})
        
        dbscan = DBSCAN(**dbscan_config)
        cluster_labels = dbscan.fit_predict(coords)
        
        # 添加聚类标签到数据
        self.df['cluster_dbscan'] = cluster_labels
        
        # 分析聚类结果
        cluster_counts = pd.Series(cluster_labels).value_counts()
        n_clusters = len(cluster_counts) - (1 if -1 in cluster_labels else 0)  # 排除噪声点
        
        # 计算每个聚类的统计信息
        cluster_stats = {}
        for cluster_id in cluster_counts.index:
            if cluster_id == -1:  # 噪声点
                cluster_stats[cluster_id] = {
                    'count': cluster_counts[cluster_id],
                    'percentage': (cluster_counts[cluster_id] / len(self.df)) * 100,
                    'type': 'noise'
                }
            else:
                cluster_data = self.df[cluster_labels == cluster_id]
                cluster_center = cluster_data[[self.lat_col, self.lon_col]].mean().values
                cluster_stats[cluster_id] = {
                    'count': len(cluster_data),
                    'percentage': (len(cluster_data) / len(self.df)) * 100,
                    'center_lat': cluster_center[0],
                    'center_lon': cluster_center[1],
                    'radius': self._calculate_cluster_radius(cluster_data, cluster_center)
                }
        
        return {
            'method': 'dbscan',
            'n_clusters': n_clusters,
            'cluster_counts': cluster_counts.to_dict(),
            'cluster_stats': cluster_stats,
            'noise_points': cluster_counts.get(-1, 0),
            'model': dbscan
        }
    
    def _calculate_cluster_radius(self, cluster_data: pd.DataFrame, center: np.ndarray) -> float:
        """计算聚类半径"""
        if len(cluster_data) == 0:
            return 0.0
        
        coords = cluster_data[[self.lat_col, self.lon_col]].values
        distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
        return distances.mean()
    
    @log_execution_time
    def analyze_hotspots(self, time_window: Optional[str] = None) -> Dict[str, Any]:
        """
        分析热点区域
        
        Args:
            time_window: 时间窗口 ('hour', 'day', 'week')
            
        Returns:
            热点分析结果
        """
        self.log_info("Analyzing hotspots")
        
        if self.lat_col is None or self.lon_col is None:
            return {'error': 'Latitude or longitude columns not found'}
        
        hotspots = {}
        
        if time_window is None:
            # 整体热点
            hotspots['overall'] = self._calculate_hotspots()
        else:
            # 按时间窗口分析热点
            if time_window in self.df.columns:
                for value in self.df[time_window].unique():
                    subset = self.df[self.df[time_window] == value]
                    hotspots[f'{time_window}_{value}'] = self._calculate_hotspots(subset)
        
        return hotspots
    
    def _calculate_hotspots(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """计算热点"""
        if data is None:
            data = self.df
        
        # 创建网格
        lat_bins = np.linspace(data[self.lat_col].min(), data[self.lat_col].max(), 50)
        lon_bins = np.linspace(data[self.lon_col].min(), data[self.lon_col].max(), 50)
        
        # 计算2D直方图
        hist, lat_edges, lon_edges = np.histogram2d(
            data[self.lat_col], data[self.lon_col], 
            bins=[lat_bins, lon_bins]
        )
        
        # 找到热点（高密度区域）
        threshold = np.percentile(hist[hist > 0], 90)  # 前10%的高密度区域
        hotspot_indices = np.where(hist >= threshold)
        
        hotspots = []
        for i, j in zip(hotspot_indices[0], hotspot_indices[1]):
            hotspots.append({
                'lat_center': (lat_edges[i] + lat_edges[i+1]) / 2,
                'lon_center': (lon_edges[j] + lon_edges[j+1]) / 2,
                'density': hist[i, j]
            })
        
        return {
            'hotspots': hotspots,
            'max_density': hist.max(),
            'mean_density': hist.mean(),
            'hotspot_count': len(hotspots)
        }
    
    @log_execution_time
    def create_spatial_visualizations(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """
        创建空间可视化
        
        Args:
            save_path: 保存路径
            
        Returns:
            图表文件路径字典
        """
        self.log_info("Creating spatial visualizations")
        
        if self.lat_col is None or self.lon_col is None:
            self.log_error("Latitude or longitude columns not found")
            return {}
        
        saved_files = {}
        
        # 1. 散点图
        plt.figure(figsize=(12, 8))
        sample_size = min(10000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        scatter = plt.scatter(sample_df[self.lon_col], sample_df[self.lat_col], 
                            c=sample_df['Base'].astype('category').cat.codes if 'Base' in sample_df.columns else 'blue',
                            s=1, alpha=0.6, cmap='tab10')
        plt.title('Geographic Distribution of Orders')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(scatter, label='Base Station')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            scatter_path = f"{save_path}/geographic_scatter.png"
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            saved_files['scatter'] = scatter_path
        
        plt.show()
        
        # 2. 热力图
        plt.figure(figsize=(12, 8))
        plt.hist2d(sample_df[self.lon_col], sample_df[self.lat_col], bins=50, cmap='Reds')
        plt.title('Order Density Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(label='Order Density')
        
        if save_path:
            heatmap_path = f"{save_path}/density_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            saved_files['heatmap'] = heatmap_path
        
        plt.show()
        
        # 3. 聚类可视化
        if 'cluster_8' in self.df.columns:
            plt.figure(figsize=(12, 8))
            for cluster_id in self.df['cluster_8'].unique():
                cluster_data = self.df[self.df['cluster_8'] == cluster_id]
                plt.scatter(cluster_data[self.lon_col], cluster_data[self.lat_col], 
                          s=1, alpha=0.6, label=f'Cluster {cluster_id}')
            
            plt.title('Geographic Clusters')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                cluster_path = f"{save_path}/geographic_clusters.png"
                plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
                saved_files['clusters'] = cluster_path
            
            plt.show()
        
        self.log_info(f"Spatial visualizations created. Saved {len(saved_files)} files.")
        return saved_files
    
    def create_interactive_map(self, save_path: Optional[str] = None) -> str:
        """
        创建交互式地图
        
        Args:
            save_path: 保存路径
            
        Returns:
            地图文件路径
        """
        self.log_info("Creating interactive map")
        
        if self.lat_col is None or self.lon_col is None:
            self.log_error("Latitude or longitude columns not found")
            return ""
        
        # 计算地图中心
        center_lat = self.df[self.lat_col].mean()
        center_lon = self.df[self.lon_col].mean()
        
        # 创建地图
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # 添加热力图
        sample_size = min(5000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        heat_data = sample_df[[self.lat_col, self.lon_col]].values.tolist()
        HeatMap(heat_data, radius=15).add_to(m)
        
        # 添加聚类中心（如果有聚类结果）
        if 'cluster_8' in self.df.columns:
            cluster_centers = self.df.groupby('cluster_8')[[self.lat_col, self.lon_col]].mean()
            
            for cluster_id, center in cluster_centers.iterrows():
                folium.Marker(
                    location=[center[self.lat_col], center[self.lon_col]],
                    popup=f'Cluster {cluster_id}',
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
        
        # 保存地图
        if save_path:
            map_path = f"{save_path}/interactive_map.html"
            m.save(map_path)
            self.log_info(f"Interactive map saved to: {map_path}")
            return map_path
        
        return m
    
    def analyze_spatial_temporal_patterns(self) -> Dict[str, Any]:
        """
        分析时空模式
        
        Returns:
            时空模式分析结果
        """
        self.log_info("Analyzing spatial-temporal patterns")
        
        if self.lat_col is None or self.lon_col is None:
            return {'error': 'Latitude or longitude columns not found'}
        
        patterns = {}
        
        # 按小时分析空间分布
        if 'hour' in self.df.columns:
            hourly_patterns = {}
            for hour in range(24):
                hour_data = self.df[self.df['hour'] == hour]
                if len(hour_data) > 0:
                    hourly_patterns[hour] = {
                        'count': len(hour_data),
                        'center_lat': hour_data[self.lat_col].mean(),
                        'center_lon': hour_data[self.lon_col].mean(),
                        'spread': hour_data[self.lat_col].std() + hour_data[self.lon_col].std()
                    }
            patterns['hourly'] = hourly_patterns
        
        # 按星期分析空间分布
        if 'weekday' in self.df.columns:
            weekday_patterns = {}
            for day in range(7):
                day_data = self.df[self.df['weekday'] == day]
                if len(day_data) > 0:
                    weekday_patterns[day] = {
                        'count': len(day_data),
                        'center_lat': day_data[self.lat_col].mean(),
                        'center_lon': day_data[self.lon_col].mean(),
                        'spread': day_data[self.lat_col].std() + day_data[self.lon_col].std()
                    }
            patterns['weekday'] = weekday_patterns
        
        return patterns
    
    def print_spatial_summary(self):
        """打印空间分析摘要"""
        analysis = self.analyze_geographic_distribution()
        
        print("=" * 80)
        print("空间分析摘要")
        print("=" * 80)
        
        if 'error' in analysis:
            print(f"❌ 错误: {analysis['error']}")
            return
        
        # 地理边界
        bounds = analysis['bounds']
        print(f"\n📍 地理边界:")
        print(f"  纬度范围: {bounds['latitude']}")
        print(f"  经度范围: {bounds['longitude']}")
        
        # 地理中心
        center = analysis['center']
        print(f"\n🎯 地理中心:")
        print(f"  平均中心: ({center['latitude']:.4f}, {center['longitude']:.4f})")
        print(f"  中位数中心: ({center['median_lat']:.4f}, {center['median_lon']:.4f})")
        
        # 密度统计
        density = analysis['density']
        print(f"\n📊 密度统计:")
        print(f"  1km范围内平均点数: {density['density_1km_mean']:.1f}")
        print(f"  5km范围内平均点数: {density['density_5km_mean']:.1f}")
        
        # 异常值
        outliers = analysis['outliers']
        print(f"\n⚠️ 地理异常值:")
        print(f"  纬度异常值: {outliers['lat_outliers_count']}")
        print(f"  经度异常值: {outliers['lon_outliers_count']}")
        print(f"  总异常值: {outliers['total_outliers']}")
        
        print("\n" + "=" * 80)

# 便捷函数
def analyze_spatial_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    快速空间模式分析
    
    Args:
        df: DataFrame
        
    Returns:
        空间分析结果
    """
    spatial_analyzer = SpatialAnalysis(df)
    return spatial_analyzer.analyze_geographic_distribution()

def create_spatial_clusters(df: pd.DataFrame, n_clusters: int = 8) -> Dict[str, Any]:
    """
    创建空间聚类
    
    Args:
        df: DataFrame
        n_clusters: 聚类数量
        
    Returns:
        聚类结果
    """
    spatial_analyzer = SpatialAnalysis(df)
    return spatial_analyzer.perform_clustering_analysis('kmeans', n_clusters=n_clusters)

def create_spatial_visualizations(df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, str]:
    """
    创建空间可视化
    
    Args:
        df: DataFrame
        save_path: 保存路径
        
    Returns:
        图表文件路径
    """
    spatial_analyzer = SpatialAnalysis(df)
    return spatial_analyzer.create_spatial_visualizations(save_path) 
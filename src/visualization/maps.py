"""
地图可视化模块

该模块提供对滴滴出行数据的地理可视化功能，包括：
- 热力图
- 聚类地图
- 路径分析
- 地理统计
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')


class MapVisualizer:
    """地图可视化类"""
    
    def __init__(self, df=None):
        """
        初始化地图可视化器
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            包含地理数据的DataFrame
        """
        self.df = df
        self.lat_column = None
        self.lng_column = None
        self.maps = {}
        
    def set_data(self, df, lat_column, lng_column):
        """
        设置地理数据
        
        Parameters:
        -----------
        df : pandas.DataFrame
            数据DataFrame
        lat_column : str
            纬度列名
        lng_column : str
            经度列名
        """
        self.df = df.copy()
        self.lat_column = lat_column
        self.lng_column = lng_column
        
        # 数据清洗
        self.df = self.df.dropna(subset=[lat_column, lng_column])
        
        # 移除异常坐标
        self.df = self.df[
            (self.df[lat_column] >= -90) & (self.df[lat_column] <= 90) &
            (self.df[lng_column] >= -180) & (self.df[lng_column] <= 180)
        ]
        
        print(f"地理数据设置完成，有效样本数量: {len(self.df)}")
        
    def create_heatmap(self, weight_column=None, radius=15, blur=10, 
                      zoom_start=10, tiles='OpenStreetMap'):
        """
        创建热力图
        
        Parameters:
        -----------
        weight_column : str, optional
            权重列名
        radius : int
            热力点半径
        blur : int
            模糊程度
        zoom_start : int
            初始缩放级别
        tiles : str
            地图瓦片类型
            
        Returns:
        --------
        folium.Map : 热力图
        """
        if self.df is None:
            raise ValueError("请先设置地理数据")
            
        # 计算地图中心
        center_lat = self.df[self.lat_column].mean()
        center_lng = self.df[self.lng_column].mean()
        
        # 创建地图
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=zoom_start,
            tiles=tiles
        )
        
        # 准备热力图数据
        heat_data = []
        for _, row in self.df.iterrows():
            weight = 1
            if weight_column:
                weight = row[weight_column]
            heat_data.append([row[self.lat_column], row[self.lng_column], weight])
        
        # 添加热力图层
        HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            max_zoom=13
        ).add_to(m)
        
        # 保存地图
        map_name = f"heatmap_{weight_column if weight_column else 'default'}"
        self.maps[map_name] = m
        
        return m
    
    def create_cluster_map(self, weight_column=None, zoom_start=10, 
                          tiles='OpenStreetMap'):
        """
        创建聚类地图
        
        Parameters:
        -----------
        weight_column : str, optional
            权重列名
        zoom_start : int
            初始缩放级别
        tiles : str
            地图瓦片类型
            
        Returns:
        --------
        folium.Map : 聚类地图
        """
        if self.df is None:
            raise ValueError("请先设置地理数据")
            
        # 计算地图中心
        center_lat = self.df[self.lat_column].mean()
        center_lng = self.df[self.lng_column].mean()
        
        # 创建地图
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=zoom_start,
            tiles=tiles
        )
        
        # 创建标记聚类
        marker_cluster = MarkerCluster().add_to(m)
        
        # 添加标记
        for _, row in self.df.iterrows():
            popup_text = f"纬度: {row[self.lat_column]:.4f}<br>经度: {row[self.lng_column]:.4f}"
            if weight_column:
                popup_text += f"<br>{weight_column}: {row[weight_column]}"
                
            folium.Marker(
                location=[row[self.lat_column], row[self.lng_column]],
                popup=popup_text,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
        
        # 保存地图
        map_name = f"cluster_map_{weight_column if weight_column else 'default'}"
        self.maps[map_name] = m
        
        return m
    
    def create_density_map(self, weight_column=None, zoom_start=10, 
                          tiles='OpenStreetMap'):
        """
        创建密度地图
        
        Parameters:
        -----------
        weight_column : str, optional
            权重列名
        zoom_start : int
            初始缩放级别
        tiles : str
            地图瓦片类型
            
        Returns:
        --------
        folium.Map : 密度地图
        """
        if self.df is None:
            raise ValueError("请先设置地理数据")
            
        # 计算地图中心
        center_lat = self.df[self.lat_column].mean()
        center_lng = self.df[self.lng_column].mean()
        
        # 创建地图
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=zoom_start,
            tiles=tiles
        )
        
        # 使用FastMarkerCluster提高性能
        locations = self.df[[self.lat_column, self.lng_column]].values.tolist()
        
        if weight_column:
            weights = self.df[weight_column].values.tolist()
            FastMarkerCluster(locations, weights=weights).add_to(m)
        else:
            FastMarkerCluster(locations).add_to(m)
        
        # 保存地图
        map_name = f"density_map_{weight_column if weight_column else 'default'}"
        self.maps[map_name] = m
        
        return m
    
    def create_choropleth_map(self, geojson_data, value_column, key_column,
                             zoom_start=10, tiles='OpenStreetMap'):
        """
        创建等值线地图
        
        Parameters:
        -----------
        geojson_data : dict
            GeoJSON数据
        value_column : str
            数值列名
        key_column : str
            地理标识列名
        zoom_start : int
            初始缩放级别
        tiles : str
            地图瓦片类型
            
        Returns:
        --------
        folium.Map : 等值线地图
        """
        if self.df is None:
            raise ValueError("请先设置地理数据")
            
        # 计算地图中心
        center_lat = self.df[self.lat_column].mean()
        center_lng = self.df[self.lng_column].mean()
        
        # 创建地图
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=zoom_start,
            tiles=tiles
        )
        
        # 添加等值线图层
        folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            data=self.df,
            columns=[key_column, value_column],
            key_on="feature.properties." + key_column,
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=value_column
        ).add_to(m)
        
        # 保存地图
        map_name = f"choropleth_map_{value_column}"
        self.maps[map_name] = m
        
        return m
    
    def plot_geographic_statistics(self, figsize=(15, 10)):
        """
        绘制地理统计图表
        
        Parameters:
        -----------
        figsize : tuple
            图表大小
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.df is None:
            raise ValueError("请先设置地理数据")
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 经纬度分布
        axes[0, 0].scatter(self.df[self.lng_column], self.df[self.lat_column], 
                          alpha=0.6, s=10)
        axes[0, 0].set_xlabel('经度')
        axes[0, 0].set_ylabel('纬度')
        axes[0, 0].set_title('地理分布散点图')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 纬度分布直方图
        axes[0, 1].hist(self.df[self.lat_column], bins=50, alpha=0.7, 
                       edgecolor='black')
        axes[0, 1].set_xlabel('纬度')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('纬度分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 经度分布直方图
        axes[1, 0].hist(self.df[self.lng_column], bins=50, alpha=0.7, 
                       edgecolor='black')
        axes[1, 0].set_xlabel('经度')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('经度分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 地理密度热力图
        # 创建网格
        lat_bins = np.linspace(self.df[self.lat_column].min(), 
                              self.df[self.lat_column].max(), 50)
        lng_bins = np.linspace(self.df[self.lng_column].min(), 
                              self.df[self.lng_column].max(), 50)
        
        density, _, _ = np.histogram2d(self.df[self.lat_column], 
                                      self.df[self.lng_column], 
                                      bins=[lat_bins, lng_bins])
        
        im = axes[1, 1].imshow(density.T, origin='lower', cmap='viridis')
        axes[1, 1].set_xlabel('纬度索引')
        axes[1, 1].set_ylabel('经度索引')
        axes[1, 1].set_title('地理密度热力图')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_spatial_patterns(self, weight_column=None):
        """
        分析空间模式
        
        Parameters:
        -----------
        weight_column : str, optional
            权重列名
            
        Returns:
        --------
        dict : 空间分析结果
        """
        if self.df is None:
            raise ValueError("请先设置地理数据")
            
        # 计算基本统计
        lat_stats = self.df[self.lat_column].describe()
        lng_stats = self.df[self.lng_column].describe()
        
        # 计算地理范围
        lat_range = self.df[self.lat_column].max() - self.df[self.lat_column].min()
        lng_range = self.df[self.lng_column].max() - self.df[self.lng_column].min()
        
        # 计算地理中心
        center_lat = self.df[self.lat_column].mean()
        center_lng = self.df[self.lng_column].mean()
        
        # 计算点密度
        total_area = lat_range * lng_range
        point_density = len(self.df) / total_area
        
        results = {
            'center': (center_lat, center_lng),
            'lat_range': lat_range,
            'lng_range': lng_range,
            'total_area': total_area,
            'point_density': point_density,
            'lat_stats': lat_stats,
            'lng_stats': lng_stats
        }
        
        # 如果有权重列，计算加权中心
        if weight_column:
            weighted_lat = np.average(self.df[self.lat_column], 
                                    weights=self.df[weight_column])
            weighted_lng = np.average(self.df[self.lng_column], 
                                    weights=self.df[weight_column])
            results['weighted_center'] = (weighted_lat, weighted_lng)
        
        # 打印结果
        print("空间模式分析结果:")
        print("-" * 50)
        print(f"地理中心: ({center_lat:.4f}, {center_lng:.4f})")
        print(f"纬度范围: {lat_range:.4f}")
        print(f"经度范围: {lng_range:.4f}")
        print(f"总面积: {total_area:.6f}")
        print(f"点密度: {point_density:.2f} 点/度²")
        
        if weight_column:
            print(f"加权中心: ({weighted_lat:.4f}, {weighted_lng:.4f})")
        
        return results
    
    def save_map(self, map_name, file_path):
        """
        保存地图到文件
        
        Parameters:
        -----------
        map_name : str
            地图名称
        file_path : str
            保存路径
        """
        if map_name not in self.maps:
            raise ValueError(f"地图 {map_name} 不存在")
            
        self.maps[map_name].save(file_path)
        print(f"地图已保存到: {file_path}")
    
    def get_map(self, map_name):
        """
        获取指定地图
        
        Parameters:
        -----------
        map_name : str
            地图名称
            
        Returns:
        --------
        folium.Map : 地图对象
        """
        if map_name not in self.maps:
            raise ValueError(f"地图 {map_name} 不存在")
            
        return self.maps[map_name]
    
    def list_maps(self):
        """
        列出所有创建的地图
        
        Returns:
        --------
        list : 地图名称列表
        """
        return list(self.maps.keys())


if __name__ == "__main__":
    # 示例用法
    print("地图可视化模块")
    print("请导入并使用 MapVisualizer 类进行地理数据可视化") 
"""
聚类分析模块

该模块提供对滴滴出行数据的聚类分析功能，包括：
- K-means聚类
- DBSCAN密度聚类
- 空间热点检测
- 聚类评估和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')


class SpatialClustering:
    """空间聚类分析类"""
    
    def __init__(self, df=None):
        """
        初始化空间聚类分析器
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            包含空间数据的DataFrame
        """
        self.df = df
        self.lat_column = None
        self.lng_column = None
        self.scaler = StandardScaler()
        self.models = {}
        
    def set_data(self, df, lat_column, lng_column, weight_column=None):
        """
        设置分析数据
        
        Parameters:
        -----------
        df : pandas.DataFrame
            数据DataFrame
        lat_column : str
            纬度列名
        lng_column : str
            经度列名
        weight_column : str, optional
            权重列名（如需求数量）
        """
        self.df = df.copy()
        self.lat_column = lat_column
        self.lng_column = lng_column
        self.weight_column = weight_column
        
        # 数据清洗
        self.df = self.df.dropna(subset=[lat_column, lng_column])
        
        # 移除异常坐标
        self.df = self.df[
            (self.df[lat_column] >= -90) & (self.df[lat_column] <= 90) &
            (self.df[lng_column] >= -180) & (self.df[lng_column] <= 180)
        ]
        
        print(f"数据清洗完成，剩余 {len(self.df)} 条记录")
        
    def prepare_features(self, include_time_features=False, include_demand_features=False):
        """
        准备聚类特征
        
        Parameters:
        -----------
        include_time_features : bool
            是否包含时间特征
        include_demand_features : bool
            是否包含需求特征
            
        Returns:
        --------
        numpy.ndarray : 特征矩阵
        """
        features = []
        
        # 基础空间特征
        coords = self.df[[self.lat_column, self.lng_column]].values
        features.append(coords)
        
        # 时间特征
        if include_time_features:
            time_features = self._extract_time_features()
            features.append(time_features)
            
        # 需求特征
        if include_demand_features and self.weight_column:
            demand_features = self.df[self.weight_column].values.reshape(-1, 1)
            features.append(demand_features)
            
        # 合并特征
        X = np.hstack(features)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def _extract_time_features(self):
        """提取时间特征"""
        # 假设有时间列，这里简化处理
        # 实际使用时需要根据具体数据调整
        time_features = np.zeros((len(self.df), 2))  # 占位符
        return time_features
    
    def kmeans_clustering(self, n_clusters=8, random_state=42, sample_size=None, **kwargs):
        """
        K-means聚类 - 优化版本
        
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        random_state : int
            随机种子
        sample_size : int, optional
            采样大小，如果数据量太大时使用
        **kwargs : dict
            其他KMeans参数
            
        Returns:
        --------
        dict : 聚类结果
        """
        # 准备特征
        X = self.prepare_features()
        
        # 如果数据量太大，进行采样
        if sample_size and len(X) > sample_size:
            print(f"数据量较大({len(X)}条)，使用{sample_size}条记录进行聚类...")
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            sample_indices = indices
        else:
            X_sample = X
            sample_indices = np.arange(len(X))
        
        # 训练模型
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
        cluster_labels_sample = kmeans.fit_predict(X_sample)
        
        # 对全量数据进行预测
        cluster_labels = kmeans.predict(X)
        
        # 计算评估指标（使用采样数据以提高速度）
        try:
            silhouette_avg = silhouette_score(X_sample, cluster_labels_sample)
            calinski_score = calinski_harabasz_score(X_sample, cluster_labels_sample)
            davies_score = davies_bouldin_score(X_sample, cluster_labels_sample)
        except Exception as e:
            print(f"评估指标计算失败: {e}")
            silhouette_avg = calinski_score = davies_score = 0
        
        # 保存结果
        result = {
            'model': kmeans,
            'labels': cluster_labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_score,
            'davies_bouldin_score': davies_score,
            'sample_size': len(X_sample),
            'total_size': len(X)
        }
        
        self.models['kmeans'] = result
        
        print(f"K-means聚类完成:")
        print(f"  聚类数量: {n_clusters}")
        print(f"  使用样本数: {len(X_sample)} / {len(X)}")
        print(f"  轮廓系数: {silhouette_avg:.4f}")
        print(f"  Calinski-Harabasz指数: {calinski_score:.4f}")
        print(f"  Davies-Bouldin指数: {davies_score:.4f}")
        
        return result
    
    def dbscan_clustering(self, eps=0.1, min_samples=5, **kwargs):
        """
        DBSCAN密度聚类
        
        Parameters:
        -----------
        eps : float
            邻域半径
        min_samples : int
            最小样本数
        **kwargs : dict
            其他DBSCAN参数
            
        Returns:
        --------
        dict : 聚类结果
        """
        # 准备特征
        X = self.prepare_features()
        
        # 训练模型
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        cluster_labels = dbscan.fit_predict(X)
        
        # 统计聚类结果
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # 计算评估指标（排除噪声点）
        if n_clusters > 1:
            mask = cluster_labels != -1
            if np.sum(mask) > 1:
                silhouette_avg = silhouette_score(X[mask], cluster_labels[mask])
                calinski_score = calinski_harabasz_score(X[mask], cluster_labels[mask])
                davies_score = davies_bouldin_score(X[mask], cluster_labels[mask])
            else:
                silhouette_avg = calinski_score = davies_score = 0
        else:
            silhouette_avg = calinski_score = davies_score = 0
        
        # 保存结果
        result = {
            'model': dbscan,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_score,
            'davies_bouldin_score': davies_score
        }
        
        self.models['dbscan'] = result
        
        print(f"DBSCAN聚类完成:")
        print(f"  聚类数量: {n_clusters}")
        print(f"  噪声点数量: {n_noise}")
        print(f"  轮廓系数: {silhouette_avg:.4f}")
        print(f"  Calinski-Harabasz指数: {calinski_score:.4f}")
        print(f"  Davies-Bouldin指数: {davies_score:.4f}")
        
        return result
    
    def hierarchical_clustering(self, n_clusters=8, linkage='ward', **kwargs):
        """
        层次聚类
        
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        linkage : str
            链接方法
        **kwargs : dict
            其他AgglomerativeClustering参数
            
        Returns:
        --------
        dict : 聚类结果
        """
        # 准备特征
        X = self.prepare_features()
        
        # 训练模型
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage, 
            **kwargs
        )
        cluster_labels = hierarchical.fit_predict(X)
        
        # 计算评估指标
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_score = calinski_harabasz_score(X, cluster_labels)
        davies_score = davies_bouldin_score(X, cluster_labels)
        
        # 保存结果
        result = {
            'model': hierarchical,
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_score,
            'davies_bouldin_score': davies_score
        }
        
        self.models['hierarchical'] = result
        
        print(f"层次聚类完成:")
        print(f"  聚类数量: {n_clusters}")
        print(f"  链接方法: {linkage}")
        print(f"  轮廓系数: {silhouette_avg:.4f}")
        print(f"  Calinski-Harabasz指数: {calinski_score:.4f}")
        print(f"  Davies-Bouldin指数: {davies_score:.4f}")
        
        return result
    
    def find_optimal_k(self, max_k=20, method='silhouette'):
        """
        寻找最优聚类数量
        
        Parameters:
        -----------
        max_k : int
            最大聚类数量
        method : str
            评估方法：'silhouette', 'elbow', 'calinski'
            
        Returns:
        --------
        dict : 最优参数和评估结果
        """
        X = self.prepare_features()
        
        k_range = range(2, max_k + 1)
        scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            if method == 'silhouette':
                score = silhouette_score(X, cluster_labels)
            elif method == 'calinski':
                score = calinski_harabasz_score(X, cluster_labels)
            elif method == 'davies':
                score = -davies_bouldin_score(X, cluster_labels)  # 越小越好，所以取负值
            else:
                score = kmeans.inertia_
                
            scores.append(score)
        
        # 找到最优k
        if method in ['silhouette', 'calinski']:
            optimal_k = k_range[np.argmax(scores)]
        else:
            optimal_k = k_range[np.argmin(scores)]
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, scores, 'bo-')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'最优k={optimal_k}')
        plt.xlabel('聚类数量 (k)')
        plt.ylabel(f'{method} 分数')
        plt.title(f'寻找最优聚类数量 - {method}方法')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'optimal_k': optimal_k,
            'k_range': list(k_range),
            'scores': scores,
            'method': method
        }
    
    def visualize_clusters(self, method='kmeans', figsize=(15, 10)):
        """
        可视化聚类结果
        
        Parameters:
        -----------
        method : str
            聚类方法：'kmeans', 'dbscan', 'hierarchical'
        figsize : tuple
            图表大小
        """
        if method not in self.models:
            raise ValueError(f"请先运行 {method} 聚类")
            
        result = self.models[method]
        labels = result['labels']
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 散点图 - 原始坐标
        scatter = axes[0, 0].scatter(
            self.df[self.lng_column], 
            self.df[self.lat_column], 
            c=labels, 
            cmap='tab10', 
            alpha=0.6
        )
        axes[0, 0].set_title(f'{method.upper()} 聚类结果')
        axes[0, 0].set_xlabel('经度')
        axes[0, 0].set_ylabel('纬度')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. 聚类中心（仅K-means）
        if method == 'kmeans' and 'centers' in result:
            centers = result['centers']
            # 反标准化中心点
            centers_original = self.scaler.inverse_transform(centers)
            axes[0, 1].scatter(
                centers_original[:, 1],  # 经度
                centers_original[:, 0],  # 纬度
                c='red', 
                marker='x', 
                s=200, 
                linewidths=3,
                label='聚类中心'
            )
            axes[0, 1].scatter(
                self.df[self.lng_column], 
                self.df[self.lat_column], 
                c=labels, 
                cmap='tab10', 
                alpha=0.3
            )
            axes[0, 1].set_title('聚类中心')
            axes[0, 1].set_xlabel('经度')
            axes[0, 1].set_ylabel('纬度')
            axes[0, 1].legend()
        
        # 3. 聚类大小分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        if -1 in unique_labels:  # DBSCAN噪声点
            noise_idx = np.where(unique_labels == -1)[0][0]
            unique_labels = np.delete(unique_labels, noise_idx)
            counts = np.delete(counts, noise_idx)
            
        axes[1, 0].bar(range(len(unique_labels)), counts, color='skyblue')
        axes[1, 0].set_title('聚类大小分布')
        axes[1, 0].set_xlabel('聚类编号')
        axes[1, 0].set_ylabel('样本数量')
        axes[1, 0].set_xticks(range(len(unique_labels)))
        axes[1, 0].set_xticklabels(unique_labels)
        
        # 4. 评估指标
        metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        metric_values = [result.get(metric, 0) for metric in metrics]
        metric_names = ['轮廓系数', 'Calinski-Harabasz', 'Davies-Bouldin']
        
        axes[1, 1].bar(metric_names, metric_values, color=['green', 'orange', 'red'])
        axes[1, 1].set_title('聚类评估指标')
        axes[1, 1].set_ylabel('分数')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for i, v in enumerate(metric_values):
            axes[1, 1].text(i, v + max(metric_values) * 0.01, f'{v:.3f}', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_heatmap(self, method='kmeans', zoom_start=10):
        """
        创建聚类热力图
        
        Parameters:
        -----------
        method : str
            聚类方法
        zoom_start : int
            地图缩放级别
            
        Returns:
        --------
        folium.Map : 热力图
        """
        if method not in self.models:
            raise ValueError(f"请先运行 {method} 聚类")
            
        result = self.models[method]
        labels = result['labels']
        
        # 计算聚类中心
        cluster_centers = []
        for label in np.unique(labels):
            if label != -1:  # 排除噪声点
                mask = labels == label
                center_lat = self.df[mask][self.lat_column].mean()
                center_lng = self.df[mask][self.lng_column].mean()
                cluster_centers.append([center_lat, center_lng])
        
        # 创建地图
        center_lat = self.df[self.lat_column].mean()
        center_lng = self.df[self.lng_column].mean()
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # 添加聚类中心
        for i, center in enumerate(cluster_centers):
            folium.Marker(
                center,
                popup=f'聚类 {i} 中心',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        # 添加热力图
        heat_data = []
        for _, row in self.df.iterrows():
            weight = 1
            if self.weight_column:
                weight = row[self.weight_column]
            heat_data.append([row[self.lat_column], row[self.lng_column], weight])
        
        HeatMap(heat_data, radius=15).add_to(m)
        
        return m
    
    def analyze_cluster_characteristics(self, method='kmeans'):
        """
        分析聚类特征
        
        Parameters:
        -----------
        method : str
            聚类方法
            
        Returns:
        --------
        pandas.DataFrame : 聚类特征分析
        """
        if method not in self.models:
            raise ValueError(f"请先运行 {method} 聚类")
            
        result = self.models[method]
        labels = result['labels']
        
        # 添加聚类标签到数据
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = labels
        
        # 分析每个聚类的特征
        cluster_analysis = []
        
        for label in np.unique(labels):
            if label == -1:  # 跳过噪声点
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == label]
            
            analysis = {
                'cluster_id': label,
                'size': len(cluster_data),
                'lat_mean': cluster_data[self.lat_column].mean(),
                'lat_std': cluster_data[self.lat_column].std(),
                'lng_mean': cluster_data[self.lng_column].mean(),
                'lng_std': cluster_data[self.lng_column].std(),
            }
            
            # 添加需求特征（如果有）
            if self.weight_column:
                analysis.update({
                    'demand_mean': cluster_data[self.weight_column].mean(),
                    'demand_std': cluster_data[self.weight_column].std(),
                    'demand_sum': cluster_data[self.weight_column].sum(),
                })
            
            cluster_analysis.append(analysis)
        
        return pd.DataFrame(cluster_analysis)
    
    def generate_clustering_report(self, output_path=None):
        """
        生成聚类分析报告
        
        Parameters:
        -----------
        output_path : str, optional
            报告保存路径
        """
        if not self.models:
            raise ValueError("请先运行聚类分析")
            
        print("=" * 60)
        print("空间聚类分析报告")
        print("=" * 60)
        
        # 数据概览
        print(f"\n1. 数据概览:")
        print(f"   总样本数: {len(self.df)}")
        print(f"   纬度范围: {self.df[self.lat_column].min():.4f} - {self.df[self.lat_column].max():.4f}")
        print(f"   经度范围: {self.df[self.lng_column].min():.4f} - {self.df[self.lng_column].max():.4f}")
        
        # 各方法结果比较
        print(f"\n2. 聚类方法比较:")
        print("-" * 50)
        
        for method, result in self.models.items():
            print(f"\n{method.upper()} 聚类:")
            print(f"  聚类数量: {len(np.unique(result['labels']))}")
            print(f"  轮廓系数: {result.get('silhouette_score', 'N/A'):.4f}")
            print(f"  Calinski-Harabasz指数: {result.get('calinski_harabasz_score', 'N/A'):.4f}")
            print(f"  Davies-Bouldin指数: {result.get('davies_bouldin_score', 'N/A'):.4f}")
            
            if method == 'dbscan':
                print(f"  噪声点数量: {result.get('n_noise', 'N/A')}")
        
        # 最优方法推荐
        best_method = max(self.models.keys(), 
                         key=lambda x: self.models[x].get('silhouette_score', 0))
        
        print(f"\n3. 推荐方法:")
        print(f"   基于轮廓系数，推荐使用: {best_method.upper()}")
        
        print("\n" + "=" * 60)
        print("报告生成完成")
        print("=" * 60)


if __name__ == "__main__":
    # 示例用法
    print("空间聚类分析模块")
    print("请导入并使用 SpatialClustering 类进行聚类分析") 
#!/usr/bin/env python3
"""
空间热点分析 - 使用框架方法
"""

import sys
sys.path.append('.')

from src.data.data_loader import DataLoader
from src.models.clustering import SpatialClustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主函数"""
    print("🚀 空间热点分析 - 使用框架方法...")
    
    # 1. 加载数据
    print("\n📊 1. 加载数据...")
    data_loader = DataLoader()
    df = data_loader.load_uber_data()
    df = data_loader.sample_data(df, sample_size=20000)  # 减少样本量，提高聚类效果
    print(f"数据形状: {df.shape}")
    
    # 2. 数据预处理
    print("\n📈 2. 数据预处理...")
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    
    # 清理坐标数据
    df = df.dropna(subset=['Lat', 'Lon'])
    df = df[
        (df['Lat'] >= 40.5) & (df['Lat'] <= 41.0) &  # 纽约大致范围
        (df['Lon'] >= -74.1) & (df['Lon'] <= -73.7)
    ]
    print(f"清理后数据形状: {df.shape}")
    
    # 3. 使用框架的空间聚类分析
    print("\n🗺️ 3. 空间聚类分析...")
    spatial_clustering = SpatialClustering()
    spatial_clustering.set_data(df, lat_column='Lat', lng_column='Lon')
    
    # 执行K-means聚类
    clustering_result = spatial_clustering.kmeans_clustering(n_clusters=8, random_state=42)
    print("✅ K-means聚类完成")
    
    # 获取聚类标签和中心
    cluster_labels = clustering_result['labels']
    centers = clustering_result['centers']
    
    # 将聚类标签添加到数据框
    df['cluster'] = cluster_labels
    
    # 4. 生成可视化
    print("\n📊 4. 生成可视化...")
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🗺️ 滴滴出行空间热点分析', fontsize=16, fontweight='bold')
    
    # 图1: 聚类结果散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['Lon'], df['Lat'], c=df['cluster'], 
                         cmap='tab10', s=10, alpha=0.6)
    ax1.set_title('空间聚类结果 (8个热点区域)', fontweight='bold')
    ax1.set_xlabel('经度')
    ax1.set_ylabel('纬度')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='聚类标签')
    
    # 图2: 聚类中心
    ax2 = axes[0, 1]
    ax2.scatter(centers[:, 1], centers[:, 0], c=range(len(centers)), 
               cmap='tab10', s=200, marker='*', edgecolors='black', linewidth=2)
    ax2.set_title('聚类中心 (热点区域中心)', fontweight='bold')
    ax2.set_xlabel('经度')
    ax2.set_ylabel('纬度')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 各聚类区域订单数量
    ax3 = axes[1, 0]
    cluster_counts = df['cluster'].value_counts().sort_index()
    bars = ax3.bar(range(len(cluster_counts)), cluster_counts.values, 
                   color=plt.cm.tab10(range(len(cluster_counts))))
    ax3.set_title('各热点区域订单数量', fontweight='bold')
    ax3.set_xlabel('热点区域')
    ax3.set_ylabel('订单数量')
    ax3.set_xticks(range(len(cluster_counts)))
    ax3.set_xticklabels([f'区域{i}' for i in cluster_counts.index])
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 图4: Base分布热力图
    ax4 = axes[1, 1]
    base_cluster = df.groupby(['cluster', 'Base']).size().unstack(fill_value=0)
    sns.heatmap(base_cluster, cmap='YlOrRd', annot=True, fmt='d', ax=ax4)
    ax4.set_title('热点区域 vs Base 分布', fontweight='bold')
    ax4.set_xlabel('Base')
    ax4.set_ylabel('热点区域')
    
    plt.tight_layout()
    plt.savefig('spatial_hotspot_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 输出分析结果
    print("\n" + "="*60)
    print("🗺️ 空间热点分析结果")
    print("="*60)
    
    print(f"📊 核心发现:")
    print(f"1. 识别出8个主要热点区域")
    print(f"2. 总订单数: {len(df):,}")
    print(f"3. 覆盖范围: 纽约市区")
    
    print(f"\n🔥 热点区域分布:")
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        center_lat = centers[cluster_id][0]
        center_lon = centers[cluster_id][1]
        print(f"   区域{cluster_id}: {count:,}次 ({percentage:.1f}%) - 中心: ({center_lat:.4f}, {center_lon:.4f})")
    
    print(f"\n🚕 Base分布分析:")
    base_counts = df['Base'].value_counts()
    print(f"   最活跃Base: {base_counts.index[0]} ({base_counts.iloc[0]:,}次)")
    print(f"   最不活跃Base: {base_counts.index[-1]} ({base_counts.iloc[-1]:,}次)")
    
    print(f"\n💡 业务建议:")
    print(f"1. 在热点区域增加车辆密度")
    print(f"2. 优化Base覆盖范围")
    print(f"3. 根据区域特点调整运营策略")
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main() 
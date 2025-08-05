#!/usr/bin/env python3
"""
滴滴出行数据分析主程序
整合所有模块，提供完整的数据分析流程
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
from config.logging_config import get_logger, PerformanceLogger
from config.settings import create_directories

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='滴滴出行数据分析平台')
    parser.add_argument('--sample-size', type=int, default=50000, 
                       help='数据采样大小 (默认: 50000)')
    parser.add_argument('--output-format', choices=['csv', 'parquet', 'pkl'], 
                       default='parquet', help='输出格式 (默认: parquet)')
    parser.add_argument('--skip-cleaning', action='store_true', 
                       help='跳过数据清洗步骤')
    parser.add_argument('--skip-features', action='store_true', 
                       help='跳过特征工程步骤')
    parser.add_argument('--verbose', action='store_true', 
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = get_logger('main_analysis')
    logger.info("=" * 60)
    logger.info("滴滴出行数据分析平台启动")
    logger.info("=" * 60)
    
    try:
        # 创建必要的目录
        create_directories()
        
        # 执行完整分析流程
        run_complete_analysis(args, logger)
        
        logger.info("=" * 60)
        logger.info("数据分析完成！")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        sys.exit(1)

def run_complete_analysis(args, logger):
    """运行完整的分析流程"""
    
    with PerformanceLogger("完整数据分析流程"):
        
        # 1. 数据加载
        logger.info("步骤 1: 数据加载")
        with PerformanceLogger("数据加载"):
            data_loader = DataLoader()
            
            # 加载原始数据
            logger.info("加载Uber数据...")
            raw_df = data_loader.load_uber_data()
            logger.info(f"原始数据形状: {raw_df.shape}")
            
            # 数据采样
            if len(raw_df) > args.sample_size:
                logger.info(f"对数据进行采样 (大小: {args.sample_size})...")
                raw_df = data_loader.sample_data(raw_df, sample_size=args.sample_size)
                logger.info(f"采样后数据形状: {raw_df.shape}")
            
            # 数据验证
            validation_results = data_loader.validate_data(raw_df)
            logger.info(f"数据验证结果: {len(validation_results['issues'])} 个问题")
            
            # 保存原始数据摘要
            data_summary = data_loader.get_data_summary(raw_df)
            logger.info(f"数据摘要: {data_summary['shape']} 行, {data_summary['memory_usage_mb']:.2f} MB")
        
        # 2. 数据清洗
        if not args.skip_cleaning:
            logger.info("步骤 2: 数据清洗")
            with PerformanceLogger("数据清洗"):
                data_cleaner = DataCleaner()
                
                # 清洗数据
                logger.info("开始数据清洗...")
                cleaned_df = data_cleaner.clean_uber_data(raw_df)
                logger.info(f"清洗后数据形状: {cleaned_df.shape}")
                
                # 创建清洗报告
                cleaning_report = data_cleaner.create_cleaning_report(raw_df, cleaned_df)
                logger.info(f"数据清洗报告:")
                logger.info(f"  - 移除行数: {cleaning_report['rows_removed']}")
                logger.info(f"  - 移除比例: {cleaning_report['removal_percentage']['rows']:.2f}%")
                logger.info(f"  - 数据质量改进: {len(cleaning_report['data_quality_improvements'])} 项")
                
                # 保存清洗后的数据
                output_filename = f"cleaned_uber_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.output_format}"
                data_loader.save_processed_data(cleaned_df, output_filename, format=args.output_format)
                logger.info(f"清洗后的数据已保存: {output_filename}")
        else:
            logger.info("跳过数据清洗步骤")
            cleaned_df = raw_df
        
        # 3. 特征工程
        if not args.skip_features:
            logger.info("步骤 3: 特征工程")
            with PerformanceLogger("特征工程"):
                feature_engineer = FeatureEngineer()
                
                # 创建所有特征
                logger.info("开始特征工程...")
                feature_df = feature_engineer.create_all_features(cleaned_df)
                logger.info(f"特征工程后数据形状: {feature_df.shape}")
                
                # 特征摘要
                feature_summary = {
                    'total_features': len(feature_df.columns),
                    'numerical_features': len(feature_df.select_dtypes(include=['number']).columns),
                    'categorical_features': len(feature_df.select_dtypes(include=['object']).columns)
                }
                logger.info(f"特征摘要:")
                logger.info(f"  - 总特征数: {feature_summary['total_features']}")
                logger.info(f"  - 数值特征: {feature_summary['numerical_features']}")
                logger.info(f"  - 分类特征: {feature_summary['categorical_features']}")
                
                # 编码分类特征
                logger.info("编码分类特征...")
                feature_df = feature_engineer.encode_categorical_features(feature_df)
                
                # 缩放数值特征
                logger.info("缩放数值特征...")
                feature_df = feature_engineer.scale_numerical_features(feature_df)
                
                # 保存特征数据
                feature_filename = f"featured_uber_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.output_format}"
                data_loader.save_processed_data(feature_df, feature_filename, format=args.output_format)
                logger.info(f"特征数据已保存: {feature_filename}")
        else:
            logger.info("跳过特征工程步骤")
            feature_df = cleaned_df
        
        # 4. 数据分析
        logger.info("步骤 4: 数据分析")
        with PerformanceLogger("数据分析"):
            # 基础统计分析
            logger.info("执行基础统计分析...")
            basic_stats = perform_basic_analysis(feature_df)
            logger.info(f"基础统计完成: {len(basic_stats)} 个指标")
            
            # 时间序列分析
            logger.info("执行时间序列分析...")
            temporal_analysis = perform_temporal_analysis(feature_df)
            logger.info(f"时间序列分析完成: {len(temporal_analysis)} 个指标")
            
            # 空间分析
            logger.info("执行空间分析...")
            spatial_analysis = perform_spatial_analysis(feature_df)
            logger.info(f"空间分析完成: {len(spatial_analysis)} 个指标")
        
        # 5. 生成报告
        logger.info("步骤 5: 生成分析报告")
        with PerformanceLogger("报告生成"):
            generate_analysis_report(
                raw_df, cleaned_df, feature_df, 
                basic_stats, temporal_analysis, spatial_analysis,
                args
            )
            logger.info("分析报告已生成")

def perform_basic_analysis(df: pd.DataFrame) -> dict:
    """执行基础统计分析"""
    stats = {}
    
    # 数据概览
    stats['total_records'] = len(df)
    stats['total_features'] = len(df.columns)
    stats['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    # 时间范围
    if 'datetime' in df.columns:
        stats['time_range_start'] = df['datetime'].min()
        stats['time_range_end'] = df['datetime'].max()
        stats['time_span_days'] = (df['datetime'].max() - df['datetime'].min()).days
    
    # 地理范围
    lat_col = 'latitude' if 'latitude' in df.columns else 'Lat'
    lon_col = 'longitude' if 'longitude' in df.columns else 'Lon'
    
    if lat_col in df.columns and lon_col in df.columns:
        stats['lat_range'] = (df[lat_col].min(), df[lat_col].max())
        stats['lon_range'] = (df[lon_col].min(), df[lon_col].max())
    
    # Base站统计
    base_col = 'base_station' if 'base_station' in df.columns else 'Base'
    if base_col in df.columns:
        stats['unique_bases'] = df[base_col].nunique()
        stats['base_distribution'] = df[base_col].value_counts().to_dict()
    
    return stats

def perform_temporal_analysis(df: pd.DataFrame) -> dict:
    """执行时间序列分析"""
    analysis = {}
    
    if 'datetime' not in df.columns:
        return analysis
    
    # 按小时统计
    if 'hour' in df.columns:
        hourly_counts = df['hour'].value_counts().sort_index()
        analysis['hourly_distribution'] = hourly_counts.to_dict()
        analysis['peak_hours'] = hourly_counts.nlargest(3).index.tolist()
        analysis['off_peak_hours'] = hourly_counts.nsmallest(3).index.tolist()
    
    # 按星期统计
    if 'weekday' in df.columns:
        weekday_counts = df['weekday'].value_counts().sort_index()
        analysis['weekday_distribution'] = weekday_counts.to_dict()
        analysis['busiest_weekday'] = weekday_counts.idxmax()
        analysis['quietest_weekday'] = weekday_counts.idxmin()
    
    # 按月份统计
    if 'month' in df.columns:
        monthly_counts = df['month'].value_counts().sort_index()
        analysis['monthly_distribution'] = monthly_counts.to_dict()
    
    # 高峰时段分析
    if 'is_peak_hour' in df.columns:
        peak_analysis = df.groupby('is_peak_hour').size()
        analysis['peak_vs_off_peak'] = peak_analysis.to_dict()
    
    return analysis

def perform_spatial_analysis(df: pd.DataFrame) -> dict:
    """执行空间分析"""
    analysis = {}
    
    lat_col = 'latitude' if 'latitude' in df.columns else 'Lat'
    lon_col = 'longitude' if 'longitude' in df.columns else 'Lon'
    
    if lat_col not in df.columns or lon_col not in df.columns:
        return analysis
    
    # 地理分布统计
    analysis['lat_stats'] = {
        'mean': df[lat_col].mean(),
        'std': df[lat_col].std(),
        'min': df[lat_col].min(),
        'max': df[lat_col].max()
    }
    
    analysis['lon_stats'] = {
        'mean': df[lon_col].mean(),
        'std': df[lon_col].std(),
        'min': df[lon_col].min(),
        'max': df[lon_col].max()
    }
    
    # 聚类分析
    if 'cluster_8' in df.columns:
        cluster_counts = df['cluster_8'].value_counts()
        analysis['cluster_distribution'] = cluster_counts.to_dict()
        analysis['most_popular_cluster'] = cluster_counts.idxmax()
        analysis['least_popular_cluster'] = cluster_counts.idxmin()
    
    # 密度分析
    if 'density_1km' in df.columns:
        analysis['density_stats'] = {
            'mean': df['density_1km'].mean(),
            'std': df['density_1km'].std(),
            'max': df['density_1km'].max()
        }
    
    return analysis

def generate_analysis_report(raw_df, cleaned_df, feature_df, 
                           basic_stats, temporal_analysis, spatial_analysis, args):
    """生成分析报告"""
    from pathlib import Path
    
    # 创建报告目录
    report_dir = Path(__file__).parent.parent / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # 生成报告文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f"analysis_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("滴滴出行数据分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析参数: 采样大小={args.sample_size}, 输出格式={args.output_format}\n")
        f.write("=" * 80 + "\n\n")
        
        # 数据概览
        f.write("1. 数据概览\n")
        f.write("-" * 40 + "\n")
        f.write(f"原始数据: {raw_df.shape[0]} 行, {raw_df.shape[1]} 列\n")
        f.write(f"清洗后数据: {cleaned_df.shape[0]} 行, {cleaned_df.shape[1]} 列\n")
        f.write(f"特征工程后: {feature_df.shape[0]} 行, {feature_df.shape[1]} 列\n")
        f.write(f"内存使用: {basic_stats.get('memory_usage_mb', 0):.2f} MB\n\n")
        
        # 基础统计
        f.write("2. 基础统计分析\n")
        f.write("-" * 40 + "\n")
        for key, value in basic_stats.items():
            if key != 'base_distribution':  # 跳过复杂的字典
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # 时间分析
        f.write("3. 时间序列分析\n")
        f.write("-" * 40 + "\n")
        f.write(f"高峰时段: {temporal_analysis.get('peak_hours', [])}\n")
        f.write(f"最繁忙的星期: {temporal_analysis.get('busiest_weekday', 'N/A')}\n")
        f.write(f"最安静的星期: {temporal_analysis.get('quietest_weekday', 'N/A')}\n")
        f.write("\n")
        
        # 空间分析
        f.write("4. 空间分析\n")
        f.write("-" * 40 + "\n")
        if 'cluster_distribution' in spatial_analysis:
            f.write(f"最热门区域: {spatial_analysis.get('most_popular_cluster', 'N/A')}\n")
            f.write(f"最冷门区域: {spatial_analysis.get('least_popular_cluster', 'N/A')}\n")
        f.write("\n")
        
        # 数据质量
        f.write("5. 数据质量评估\n")
        f.write("-" * 40 + "\n")
        f.write(f"缺失值: {feature_df.isnull().sum().sum()}\n")
        f.write(f"重复行: {feature_df.duplicated().sum()}\n")
        f.write(f"唯一值比例: {feature_df.nunique().mean() / len(feature_df):.2%}\n")
        f.write("\n")
        
        # 建议
        f.write("6. 分析建议\n")
        f.write("-" * 40 + "\n")
        f.write("1. 重点关注高峰时段的需求预测\n")
        f.write("2. 优化热门区域的车辆调度\n")
        f.write("3. 考虑时间-空间联合建模\n")
        f.write("4. 定期更新模型以适应季节性变化\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"分析报告已生成: {report_file}")

if __name__ == "__main__":
    main() 
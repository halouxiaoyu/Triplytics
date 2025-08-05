#!/usr/bin/env python3
"""
使用框架方法的时间模式分析 - 增强版
"""

import sys
sys.path.append('.')

from src.data.data_loader import DataLoader
from src.analysis.temporal_analysis import TemporalAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主函数"""
    print("🚀 使用框架方法进行时间模式分析 - 增强版...")
    
    # 1. 使用封装的数据加载器
    print("\n📊 1. 加载数据...")
    data_loader = DataLoader()
    df = data_loader.load_uber_data()
    df = data_loader.sample_data(df, sample_size=50000)
    print(f"数据形状: {df.shape}")
    
    # 2. 使用封装的时间分析器
    print("\n📈 2. 时间模式分析...")
    temporal_analyzer = TemporalAnalysis()
    
    # 时间字段处理
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df['hour'] = df['Date/Time'].dt.hour
    df['weekday'] = df['Date/Time'].dt.weekday
    df['day'] = df['Date/Time'].dt.day
    df['month'] = df['Date/Time'].dt.month
    df['is_weekend'] = df['weekday'].apply(lambda x: '周末' if x >= 5 else '工作日')
    
    # 添加一个虚拟的value_column用于分析（每行都是1，表示一次出行）
    df['trip_count'] = 1
    
    # 设置数据到分析器
    temporal_analyzer.set_data(df, time_column='Date/Time', value_column='trip_count')
    
    # 3. 分析每小时模式
    print("\n⏰ 3. 分析每小时模式...")
    hourly_analysis = temporal_analyzer.analyze_hourly_patterns()
    print("✅ 每小时模式分析完成")
    
    # 4. 分析工作日vs周末
    print("\n📅 4. 分析工作日vs周末...")
    weekday_hourly = df.groupby(['is_weekend', 'hour']).size().reset_index(name='count')
    weekday_avg = weekday_hourly[weekday_hourly['is_weekend'] == '工作日']['count'].mean()
    weekend_avg = weekday_hourly[weekday_hourly['is_weekend'] == '周末']['count'].mean()
    print("✅ 工作日vs周末分析完成")
    
    # 5. 分析高峰时段
    print("\n🚦 5. 分析高峰时段...")
    df['peak_period'] = df['hour'].apply(lambda x: '早高峰' if 7 <= x <= 9 else 
                                        ('晚高峰' if 17 <= x <= 19 else 
                                         ('深夜时段' if 2 <= x <= 5 else '其他时段')))
    peak_counts = df['peak_period'].value_counts()
    print("✅ 高峰时段分析完成")
    
    # 6. 生成丰富的可视化
    print("\n📊 6. 生成丰富的可视化...")
    
    # 创建子图布局
    fig = plt.figure(figsize=(20, 15))
    
    # 图1: 每小时分布
    plt.subplot(3, 3, 1)
    hourly_counts = df['hour'].value_counts().sort_index()
    bars = plt.bar(hourly_counts.index, hourly_counts.values, color='skyblue', alpha=0.7)
    plt.title('每小时出行量分布', fontsize=12, fontweight='bold')
    plt.xlabel('小时')
    plt.ylabel('出行次数')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    
    # 标记高峰时段
    peak_hours = [7, 8, 9, 17, 18, 19]
    for hour in peak_hours:
        if hour in hourly_counts.index:
            plt.bar(hour, hourly_counts[hour], color='red', alpha=0.8)
    
    # 图2: 工作日vs周末对比
    plt.subplot(3, 3, 2)
    for day_type in ['工作日', '周末']:
        data = weekday_hourly[weekday_hourly['is_weekend'] == day_type]
        plt.plot(data['hour'], data['count'], marker='o', label=day_type, linewidth=2)
    plt.title('工作日 vs 周末 出行趋势', fontsize=12, fontweight='bold')
    plt.xlabel('小时')
    plt.ylabel('出行次数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 图3: 高峰时段分布
    plt.subplot(3, 3, 3)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    plt.pie(peak_counts.values, labels=peak_counts.index, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    plt.title('高峰时段占比', fontsize=12, fontweight='bold')
    
    # 图4: 星期几分布
    plt.subplot(3, 3, 4)
    weekday_counts = df['weekday'].value_counts().sort_index()
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    plt.bar(weekday_names, weekday_counts.values, color='lightcoral', alpha=0.8)
    plt.title('星期几出行量分布', fontsize=12, fontweight='bold')
    plt.xlabel('星期')
    plt.ylabel('出行次数')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 图5: 每日趋势
    plt.subplot(3, 3, 5)
    daily_counts = df.groupby('day').size()
    plt.plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2, color='green')
    plt.title('每日出行量趋势', fontsize=12, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('出行次数')
    plt.grid(True, alpha=0.3)
    
    # 图6: 热力图 - 小时vs星期
    plt.subplot(3, 3, 6)
    heatmap_data = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, cbar_kws={'label': '出行次数'})
    plt.title('星期几 vs 小时 热力图', fontsize=12, fontweight='bold')
    plt.xlabel('小时')
    plt.ylabel('星期几')
    
    # 图7: 高峰时段对比
    plt.subplot(3, 3, 7)
    peak_period_hourly = df.groupby(['peak_period', 'hour']).size().unstack(fill_value=0)
    peak_period_hourly.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'red', 'orange', 'green'])
    plt.title('高峰时段每小时分布', fontsize=12, fontweight='bold')
    plt.xlabel('高峰时段')
    plt.ylabel('出行次数')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 图8: 工作日vs周末占比
    plt.subplot(3, 3, 8)
    day_type_counts = df['is_weekend'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    plt.pie(day_type_counts.values, labels=day_type_counts.index, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    plt.title('工作日 vs 周末 总体分布', fontsize=12, fontweight='bold')
    
    # 图9: 统计摘要
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # 创建统计摘要文本
    stats_text = f"""
统计摘要:
• 总出行次数: {len(df):,}
• 最高峰时段: {hourly_counts.idxmax()}时 ({hourly_counts.max():,}次)
• 最低谷时段: {hourly_counts.idxmin()}时 ({hourly_counts.min():,}次)
• 工作日平均: {weekday_avg:.0f}次/小时
• 周末平均: {weekend_avg:.0f}次/小时
• 差异倍数: {weekday_avg/weekend_avg:.2f}
• 晚高峰占比: {(peak_counts.get('晚高峰', 0)/len(df)*100):.1f}%
• 早高峰占比: {(peak_counts.get('早高峰', 0)/len(df)*100):.1f}%
    """
    
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 生成额外的详细分析图表
    print("\n📈 7. 生成详细分析图表...")
    
    # 创建第二个图表集
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 每小时分布（更详细）
    ax1 = axes[0, 0]
    bars = ax1.bar(hourly_counts.index, hourly_counts.values, color='skyblue', alpha=0.7)
    ax1.set_title('每小时出行量详细分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('小时 (0-23)')
    ax1.set_ylabel('出行次数')
    ax1.set_xticks(range(0, 24))
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    
    # 子图2: 工作日vs周末对比（更详细）
    ax2 = axes[0, 1]
    for day_type in ['工作日', '周末']:
        data = weekday_hourly[weekday_hourly['is_weekend'] == day_type]
        ax2.plot(data['hour'], data['count'], marker='o', label=day_type, linewidth=2, markersize=6)
    ax2.set_title('工作日 vs 周末 详细对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('小时 (0-23)')
    ax2.set_ylabel('出行次数')
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 星期几分布（更详细）
    ax3 = axes[1, 0]
    bars = ax3.bar(weekday_names, weekday_counts.values, color='lightcoral', alpha=0.8)
    ax3.set_title('星期几出行量详细分布', fontsize=14, fontweight='bold')
    ax3.set_xlabel('星期')
    ax3.set_ylabel('出行次数')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 子图4: 高峰时段分析
    ax4 = axes[1, 1]
    peak_period_data = df.groupby('peak_period').size()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax4.bar(peak_period_data.index, peak_period_data.values, color=colors, alpha=0.8)
    ax4.set_title('高峰时段详细分析', fontsize=14, fontweight='bold')
    ax4.set_ylabel('出行次数')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 添加百分比标签
    for bar in bars:
        height = bar.get_height()
        percentage = (height / len(df)) * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. 输出详细结果
    print("\n" + "="*60)
    print("🚗 框架方法时间模式分析结果 - 增强版")
    print("="*60)
    
    max_hour = hourly_counts.idxmax()
    min_hour = hourly_counts.idxmin()
    
    print(f"📊 核心发现:")
    print(f"1. 高峰时段: 早高峰(7-9点), 晚高峰(17-19点)")
    print(f"2. 最高峰: {max_hour}时 ({hourly_counts[max_hour]:,}次)")
    print(f"3. 最低谷: {min_hour}时 ({hourly_counts[min_hour]:,}次)")
    print(f"4. 工作日平均: {weekday_avg:.0f}次/小时")
    print(f"5. 周末平均: {weekend_avg:.0f}次/小时")
    print(f"6. 差异倍数: {weekday_avg/weekend_avg:.2f}")
    
    print(f"\n🚦 高峰时段分布:")
    for period, count in peak_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {period}: {count:,}次 ({percentage:.1f}%)")
    
    print(f"\n📅 星期几分析:")
    for i, (day, count) in enumerate(weekday_counts.items()):
        percentage = (count / len(df)) * 100
        print(f"   {weekday_names[i]}: {count:,}次 ({percentage:.1f}%)")
    
    print(f"\n💡 业务建议:")
    print(f"1. 在{max_hour}时增加车辆调度")
    print(f"2. 在{min_hour}时减少车辆投放")
    print(f"3. 针对工作日和周末制定不同策略")
    print(f"4. 重点关注晚高峰时段的服务保障")
    print(f"5. 根据星期几的分布调整运营策略")
    
    print(f"\n🔧 使用的框架组件:")
    print(f"1. DataLoader - 数据加载和采样")
    print(f"2. TemporalAnalysis - 时间序列分析")
    print(f"3. 封装的可视化方法")
    print(f"4. 日志系统 - 执行过程记录")
    
    print(f"\n📁 生成的文件:")
    print(f"1. comprehensive_time_analysis.png - 综合分析图表")
    print(f"2. detailed_time_analysis.png - 详细分析图表")
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main() 
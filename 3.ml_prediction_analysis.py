#!/usr/bin/env python3
"""
机器学习预测价值分析 - 滴滴出行需求预测
"""

import sys
sys.path.append('.')

from src.data.data_loader import DataLoader
from src.models.prediction import DemandPredictor
from src.models.evaluation import ModelEvaluator
from src.visualization.charts import ChartGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主函数"""
    print("🤖 机器学习预测价值分析 - 滴滴出行需求预测...")
    
    # 1. 加载数据
    print("\n📊 1. 加载和预处理数据...")
    data_loader = DataLoader()
    df = data_loader.load_uber_data()
    df = data_loader.sample_data(df, sample_size=50000)  # 增加样本量以提高模型性能
    print(f"数据形状: {df.shape}")
    
    # 2. 数据预处理
    print("\n📈 2. 数据预处理...")
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df = df.dropna(subset=['Lat', 'Lon', 'Base'])
    
    # 添加时间特征
    df['hour'] = df['Date/Time'].dt.hour
    df['day_of_week'] = df['Date/Time'].dt.dayofweek
    df['month'] = df['Date/Time'].dt.month
    df['date'] = df['Date/Time'].dt.date
    
    print(f"预处理后数据形状: {df.shape}")
    
    # 3. 需求预测模型训练和评估
    print("\n🤖 3. 需求预测模型训练和评估...")
    
    # 创建需求预测器
    demand_predictor = DemandPredictor()
    
    # 准备特征数据
    feature_columns = ['hour', 'day_of_week', 'month', 'Lat', 'Lon']
    
    # 创建需求预测目标变量（按小时聚合订单量）
    df['hour_date'] = df['Date/Time'].dt.floor('H')
    hourly_demand = df.groupby('hour_date').size().reset_index(name='demand')
    hourly_demand['hour'] = hourly_demand['hour_date'].dt.hour
    hourly_demand['day_of_week'] = hourly_demand['hour_date'].dt.dayofweek
    hourly_demand['month'] = hourly_demand['hour_date'].dt.month
    hourly_demand['date'] = hourly_demand['hour_date'].dt.date
    
    # 添加空间特征（使用平均坐标）
    spatial_features = df.groupby(df['Date/Time'].dt.floor('H')).agg({
        'Lat': 'mean',
        'Lon': 'mean'
    }).reset_index()
    spatial_features.columns = ['hour_date', 'Lat', 'Lon']
    
    # 合并数据
    prediction_df = hourly_demand.merge(spatial_features, on='hour_date')
    target_column = 'demand'  # 使用小时订单量作为目标变量
    
    # 设置数据
    demand_predictor.set_data(prediction_df, feature_columns, target_column)
    
    # 训练多个模型
    models = {
        'RandomForest': demand_predictor.train_random_forest,
        'XGBoost': demand_predictor.train_xgboost,
        'LightGBM': demand_predictor.train_lightgbm
    }
    
    model_results = {}
    
    for model_name, train_func in models.items():
        print(f"\n训练 {model_name} 模型...")
        try:
            result = train_func()
            
            # 提取结果
            model = result['model']
            y_test = prediction_df[target_column].iloc[-int(len(prediction_df)*0.2):]  # 使用最后20%作为测试集
            y_pred = result['predictions']['test']
            metrics = result['metrics']
            
            model_results[model_name] = {
                'model': model,
                'metrics': {
                    'r2_score': metrics['test_r2'],
                    'mae': metrics['test_mae'],
                    'rmse': metrics['test_rmse']
                },
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"✅ {model_name} 训练完成")
            print(f"   R² = {metrics['test_r2']:.4f}")
            print(f"   MAE = {metrics['test_mae']:.2f}")
            print(f"   RMSE = {metrics['test_rmse']:.2f}")
            
        except Exception as e:
            print(f"❌ {model_name} 训练失败: {e}")
    
    # 4. 时间序列预测
    print("\n📅 4. 时间序列预测分析...")
    
    # 按日期聚合订单量
    daily_demand = df.groupby('date').size().reset_index(name='demand')
    daily_demand['date'] = pd.to_datetime(daily_demand['date'])
    daily_demand = daily_demand.sort_values('date')
    
    # 训练Prophet时间序列模型
    try:
        # 准备Prophet数据
        prophet_df = daily_demand.copy()
        prophet_df.columns = ['ds', 'y']
        
        # 使用Prophet进行预测
        from prophet import Prophet
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        prophet_model.fit(prophet_df)
        
        # 预测未来7天
        future = prophet_model.make_future_dataframe(periods=7)
        forecast = prophet_model.predict(future)
        
        # 提取预测结果
        arima_forecast = forecast['yhat'].tail(7).values
        future_dates = forecast['ds'].tail(7).dt.to_pydatetime()
        
        print("✅ Prophet时间序列模型训练完成")
        print(f"   预测未来7天订单量趋势")
        
    except Exception as e:
        print(f"❌ Prophet模型训练失败: {e}")
        arima_forecast = None
        future_dates = []
    
    # 5. 生成可视化
    print("\n📊 5. 生成可视化分析...")
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🤖 机器学习预测价值分析 - 滴滴出行需求预测', fontsize=16, fontweight='bold')
    
    # 图1: 模型性能对比
    ax1 = axes[0, 0]
    if model_results:
        model_names = list(model_results.keys())
        r2_scores = [model_results[name]['metrics']['r2_score'] for name in model_names]
        mae_scores = [model_results[name]['metrics']['mae'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='skyblue', alpha=0.8)
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width/2, mae_scores, width, label='MAE', color='lightcoral', alpha=0.8)
        
        ax1.set_title('模型性能对比', fontweight='bold')
        ax1.set_xlabel('模型')
        ax1.set_ylabel('R² Score', color='skyblue')
        ax1_twin.set_ylabel('MAE', color='lightcoral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                    f'{r2_scores[i]:.3f}', ha='center', va='bottom', fontweight='bold')
            ax1_twin.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                         f'{mae_scores[i]:.1f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, '模型训练失败\n请检查数据格式', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('模型性能对比', fontweight='bold')
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='skyblue', alpha=0.8)
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, mae_scores, width, label='MAE', color='lightcoral', alpha=0.8)
    
    ax1.set_title('模型性能对比', fontweight='bold')
    ax1.set_xlabel('模型')
    ax1.set_ylabel('R² Score', color='skyblue')
    ax1_twin.set_ylabel('MAE', color='lightcoral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                f'{r2_scores[i]:.3f}', ha='center', va='bottom', fontweight='bold')
        ax1_twin.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                     f'{mae_scores[i]:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 图2: 预测vs实际值散点图（使用最佳模型）
    ax2 = axes[0, 1]
    if model_results:
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics']['r2_score'])
        best_result = model_results[best_model]
        
        ax2.scatter(best_result['y_test'], best_result['y_pred'], alpha=0.6, color='green')
        ax2.plot([best_result['y_test'].min(), best_result['y_test'].max()], 
                 [best_result['y_test'].min(), best_result['y_test'].max()], 
                 'r--', lw=2, label='完美预测线')
        ax2.set_title(f'{best_model} 预测vs实际值', fontweight='bold')
        ax2.set_xlabel('实际值')
        ax2.set_ylabel('预测值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '无可用模型结果', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('预测vs实际值', fontweight='bold')
    
    # 图3: 时间序列预测
    ax3 = axes[1, 0]
    if arima_forecast is not None:
        # 绘制历史数据
        ax3.plot(daily_demand['date'], daily_demand['demand'], 
                label='历史数据', color='blue', linewidth=2)
        
        # 绘制预测数据
        ax3.plot(future_dates, arima_forecast, 
                label='Prophet预测', color='red', linewidth=2, linestyle='--')
        
        ax3.set_title('时间序列预测 (未来7天)', fontweight='bold')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('订单量')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    
    # 图4: 特征重要性（使用最佳模型）
    ax4 = axes[1, 1]
    if model_results and hasattr(best_result['model'], 'feature_importances_'):
        importances = best_result['model'].feature_importances_
        feature_names = feature_columns
        
        # 排序特征重要性
        indices = np.argsort(importances)[::-1]
        
        ax4.bar(range(len(importances)), importances[indices], color='orange', alpha=0.8)
        ax4.set_title(f'{best_model} 特征重要性', fontweight='bold')
        ax4.set_xlabel('特征')
        ax4.set_ylabel('重要性')
        ax4.set_xticks(range(len(importances)))
        ax4.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '无特征重要性数据', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('特征重要性', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ml_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 输出分析结果
    print("\n" + "="*80)
    print("🤖 机器学习预测价值分析结果")
    print("="*80)
    
    print(f"📊 模型性能总结:")
    print(f"数据集大小: {len(prediction_df):,} 条记录")
    print(f"特征数量: {len(feature_columns)} 个")
    
    if model_results:
        print(f"测试集大小: {len(next(iter(model_results.values()))['y_test']):,} 条记录")
        
        print(f"\n🏆 模型排名 (按R²分数):")
        sorted_models = sorted(model_results.items(), 
                              key=lambda x: x[1]['metrics']['r2_score'], 
                              reverse=True)
        
        for i, (model_name, result) in enumerate(sorted_models, 1):
            metrics = result['metrics']
            print(f"   {i}. {model_name}:")
            print(f"      R² = {metrics['r2_score']:.4f}")
            print(f"      MAE = {metrics['mae']:.2f}")
            print(f"      RMSE = {metrics['rmse']:.2f}")
    else:
        print("❌ 所有模型训练失败")
    
    print(f"\n📈 时间序列预测:")
    if arima_forecast is not None:
        print(f"   Prophet模型成功预测未来7天订单量趋势")
        print(f"   预测趋势: {'上升' if arima_forecast[-1] > arima_forecast[0] else '下降'}")
        print(f"   预测范围: {arima_forecast.min():.0f} - {arima_forecast.max():.0f} 订单/天")
    
    print(f"\n💡 业务价值:")
    if model_results:
        best_r2 = max([result['metrics']['r2_score'] for result in model_results.values()])
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['metrics']['r2_score'])
        print(f"1. 高精度预测: 最佳模型R²达到{best_r2:.4f}")
        print(f"2. 需求规划: 可提前预测订单量，优化车辆调度")
        print(f"3. 成本控制: 减少空驶率，提高运营效率")
        print(f"4. 用户体验: 缩短等待时间，提升服务质量")
        
        print(f"\n🚀 应用建议:")
        print(f"1. 部署{best_model_name}模型到生产环境")
        print(f"2. 实时更新模型参数")
        print(f"3. 结合时间序列预测进行动态调度")
        print(f"4. 建立A/B测试验证模型效果")
    else:
        print("1. 需要修复数据格式问题")
        print("2. 检查特征工程过程")
        print("3. 验证模型训练参数")
        print("4. 确保数据质量")
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main() 
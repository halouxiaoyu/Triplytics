"""
时间序列分析模块

该模块提供对滴滴出行数据的时间序列分析功能，包括：
- 时间模式分析
- 季节性分析
- 趋势分析
- 周期性检测
- 时间序列预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TemporalAnalysis:
    """时间序列分析类"""
    
    def __init__(self, df=None):
        """
        初始化时间序列分析器
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            包含时间序列数据的DataFrame
        """
        self.df = df
        self.time_column = None
        self.value_column = None
        
    def set_data(self, df, time_column, value_column):
        """
        设置分析数据
        
        Parameters:
        -----------
        df : pandas.DataFrame
            数据DataFrame
        time_column : str
            时间列名
        value_column : str
            数值列名
        """
        self.df = df.copy()
        self.time_column = time_column
        self.value_column = value_column
        
        # 确保时间列格式正确
        if not pd.api.types.is_datetime64_any_dtype(self.df[time_column]):
            self.df[time_column] = pd.to_datetime(self.df[time_column])
            
        # 按时间排序
        self.df = self.df.sort_values(time_column).reset_index(drop=True)
        
    def analyze_hourly_patterns(self, group_by='hour'):
        """
        分析小时级模式
        
        Parameters:
        -----------
        group_by : str
            分组方式：'hour', 'day', 'week', 'month'
            
        Returns:
        --------
        dict : 包含各种时间模式的统计信息
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 提取时间特征
        df_analysis = self.df.copy()
        df_analysis['hour'] = df_analysis[self.time_column].dt.hour
        df_analysis['day'] = df_analysis[self.time_column].dt.day
        df_analysis['weekday'] = df_analysis[self.time_column].dt.dayofweek
        df_analysis['week'] = df_analysis[self.time_column].dt.isocalendar().week
        df_analysis['month'] = df_analysis[self.time_column].dt.month
        
        patterns = {}
        
        if group_by == 'hour':
            hourly_stats = df_analysis.groupby('hour')[self.value_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            patterns['hourly'] = hourly_stats
            
        elif group_by == 'day':
            daily_stats = df_analysis.groupby('day')[self.value_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            patterns['daily'] = daily_stats
            
        elif group_by == 'week':
            weekly_stats = df_analysis.groupby('week')[self.value_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            patterns['weekly'] = weekly_stats
            
        elif group_by == 'month':
            monthly_stats = df_analysis.groupby('month')[self.value_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            patterns['monthly'] = monthly_stats
            
        return patterns
    
    def plot_hourly_patterns(self, figsize=(15, 10)):
        """
        绘制小时级模式图表
        
        Parameters:
        -----------
        figsize : tuple
            图表大小
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        df_analysis = self.df.copy()
        df_analysis['hour'] = df_analysis[self.time_column].dt.hour
        df_analysis['weekday'] = df_analysis[self.time_column].dt.dayofweek
        df_analysis['weekday_name'] = df_analysis[self.time_column].dt.day_name()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 24小时模式
        hourly_avg = df_analysis.groupby('hour')[self.value_column].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
        axes[0, 0].set_title('24小时需求模式')
        axes[0, 0].set_xlabel('小时')
        axes[0, 0].set_ylabel('平均需求')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 工作日vs周末
        weekday_avg = df_analysis.groupby('weekday')[self.value_column].mean()
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        axes[0, 1].bar(weekday_names, weekday_avg.values, color='skyblue')
        axes[0, 1].set_title('工作日需求模式')
        axes[0, 1].set_xlabel('星期')
        axes[0, 1].set_ylabel('平均需求')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 热力图：小时 x 星期
        pivot_table = df_analysis.pivot_table(
            values=self.value_column, 
            index='weekday_name', 
            columns='hour', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 0])
        axes[1, 0].set_title('需求热力图 (星期 x 小时)')
        
        # 4. 时间序列趋势
        daily_avg = df_analysis.groupby(df_analysis[self.time_column].dt.date)[self.value_column].mean()
        axes[1, 1].plot(daily_avg.index, daily_avg.values, linewidth=1)
        axes[1, 1].set_title('日需求趋势')
        axes[1, 1].set_xlabel('日期')
        axes[1, 1].set_ylabel('平均需求')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def seasonal_decomposition(self, period=24, extrapolate_trend='freq'):
        """
        时间序列分解
        
        Parameters:
        -----------
        period : int
            季节性周期
        extrapolate_trend : str
            趋势外推方法
            
        Returns:
        --------
        dict : 分解结果
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 重采样到小时级别
        df_resampled = self.df.set_index(self.time_column)[self.value_column].resample('H').mean()
        df_resampled = df_resampled.fillna(method='ffill')
        
        # 执行分解
        decomposition = seasonal_decompose(
            df_resampled, 
            period=period, 
            extrapolate_trend=extrapolate_trend
        )
        
        # 绘制分解结果
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition.observed.plot(ax=axes[0], title='原始数据')
        decomposition.trend.plot(ax=axes[1], title='趋势')
        decomposition.seasonal.plot(ax=axes[2], title='季节性')
        decomposition.resid.plot(ax=axes[3], title='残差')
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
        return {
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'resid': decomposition.resid
        }
    
    def stationarity_test(self):
        """
        平稳性检验
        
        Returns:
        --------
        dict : 检验结果
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 重采样到小时级别
        df_resampled = self.df.set_index(self.time_column)[self.value_column].resample('H').mean()
        df_resampled = df_resampled.fillna(method='ffill')
        
        # ADF检验
        adf_result = adfuller(df_resampled.dropna())
        
        # KPSS检验
        kpss_result = kpss(df_resampled.dropna())
        
        results = {
            'adf': {
                'statistic': adf_result[0],
                'pvalue': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            },
            'kpss': {
                'statistic': kpss_result[0],
                'pvalue': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
        }
        
        print("平稳性检验结果:")
        print(f"ADF检验 - 统计量: {results['adf']['statistic']:.4f}, p值: {results['adf']['pvalue']:.4f}")
        print(f"ADF检验 - 是否平稳: {results['adf']['is_stationary']}")
        print(f"KPSS检验 - 统计量: {results['kpss']['statistic']:.4f}, p值: {results['kpss']['pvalue']:.4f}")
        print(f"KPSS检验 - 是否平稳: {results['kpss']['is_stationary']}")
        
        return results
    
    def autocorrelation_analysis(self, lags=50):
        """
        自相关分析
        
        Parameters:
        -----------
        lags : int
            滞后阶数
            
        Returns:
        --------
        tuple : (acf, pacf) 自相关和偏自相关函数
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 重采样到小时级别
        df_resampled = self.df.set_index(self.time_column)[self.value_column].resample('H').mean()
        df_resampled = df_resampled.fillna(method='ffill')
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # ACF图
        plot_acf(df_resampled.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('自相关函数 (ACF)')
        
        # PACF图
        plot_pacf(df_resampled.dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('偏自相关函数 (PACF)')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def arima_forecast(self, order=(1, 1, 1), forecast_steps=24):
        """
        ARIMA模型预测
        
        Parameters:
        -----------
        order : tuple
            ARIMA参数 (p, d, q)
        forecast_steps : int
            预测步数
            
        Returns:
        --------
        dict : 预测结果
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 重采样到小时级别
        df_resampled = self.df.set_index(self.time_column)[self.value_column].resample('H').mean()
        df_resampled = df_resampled.fillna(method='ffill')
        
        # 训练ARIMA模型
        model = ARIMA(df_resampled.dropna(), order=order)
        fitted_model = model.fit()
        
        # 预测
        forecast = fitted_model.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(
            start=df_resampled.index[-1] + pd.Timedelta(hours=1),
            periods=forecast_steps,
            freq='H'
        )
        
        # 绘制结果
        plt.figure(figsize=(15, 6))
        plt.plot(df_resampled.index[-100:], df_resampled.values[-100:], label='历史数据')
        plt.plot(forecast_index, forecast, label='预测', color='red')
        plt.fill_between(forecast_index, 
                        forecast - 1.96 * fitted_model.params[-1],
                        forecast + 1.96 * fitted_model.params[-1],
                        alpha=0.3, color='red', label='95%置信区间')
        plt.title(f'ARIMA{order} 预测结果')
        plt.xlabel('时间')
        plt.ylabel('需求')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'model': fitted_model,
            'forecast': forecast,
            'forecast_index': forecast_index,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
    
    def prophet_forecast(self, forecast_periods=24, changepoint_prior_scale=0.05):
        """
        Prophet模型预测
        
        Parameters:
        -----------
        forecast_periods : int
            预测期数
        changepoint_prior_scale : float
            变点先验尺度
            
        Returns:
        --------
        dict : 预测结果
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 准备Prophet数据格式
        df_prophet = self.df.copy()
        df_prophet = df_prophet.rename(columns={self.time_column: 'ds', self.value_column: 'y'})
        
        # 重采样到小时级别
        df_prophet = df_prophet.set_index('ds')['y'].resample('H').mean().reset_index()
        df_prophet = df_prophet.fillna(method='ffill')
        
        # 训练Prophet模型
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        model.fit(df_prophet)
        
        # 预测
        future = model.make_future_dataframe(periods=forecast_periods, freq='H')
        forecast = model.predict(future)
        
        # 绘制结果
        fig = model.plot(forecast)
        plt.title('Prophet 预测结果')
        plt.show()
        
        # 绘制组件
        fig2 = model.plot_components(forecast)
        plt.show()
        
        return {
            'model': model,
            'forecast': forecast,
            'components': model.plot_components(forecast)
        }
    
    def compare_models(self, test_size=0.2):
        """
        比较不同预测模型
        
        Parameters:
        -----------
        test_size : float
            测试集比例
            
        Returns:
        --------
        dict : 模型比较结果
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 重采样到小时级别
        df_resampled = self.df.set_index(self.time_column)[self.value_column].resample('H').mean()
        df_resampled = df_resampled.fillna(method='ffill')
        
        # 分割数据
        split_idx = int(len(df_resampled) * (1 - test_size))
        train = df_resampled[:split_idx]
        test = df_resampled[split_idx:]
        
        results = {}
        
        # ARIMA模型
        try:
            arima_model = ARIMA(train, order=(1, 1, 1))
            arima_fitted = arima_model.fit()
            arima_pred = arima_fitted.forecast(steps=len(test))
            
            arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
            arima_mae = mean_absolute_error(test, arima_pred)
            
            results['ARIMA'] = {
                'rmse': arima_rmse,
                'mae': arima_mae,
                'predictions': arima_pred
            }
        except Exception as e:
            print(f"ARIMA模型训练失败: {e}")
        
        # Prophet模型
        try:
            train_prophet = train.reset_index().rename(columns={train.name: 'y'})
            train_prophet.columns = ['ds', 'y']
            
            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            prophet_model.fit(train_prophet)
            
            future = prophet_model.make_future_dataframe(periods=len(test), freq='H')
            prophet_forecast = prophet_model.predict(future)
            prophet_pred = prophet_forecast.iloc[-len(test):]['yhat']
            
            prophet_rmse = np.sqrt(mean_squared_error(test, prophet_pred))
            prophet_mae = mean_absolute_error(test, prophet_pred)
            
            results['Prophet'] = {
                'rmse': prophet_rmse,
                'mae': prophet_mae,
                'predictions': prophet_pred
            }
        except Exception as e:
            print(f"Prophet模型训练失败: {e}")
        
        # 绘制比较结果
        plt.figure(figsize=(15, 6))
        plt.plot(test.index, test.values, label='实际值', linewidth=2)
        
        for model_name, result in results.items():
            plt.plot(test.index, result['predictions'], label=f'{model_name}预测', alpha=0.7)
        
        plt.title('模型预测比较')
        plt.xlabel('时间')
        plt.ylabel('需求')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 打印评估指标
        print("\n模型性能比较:")
        print("-" * 50)
        for model_name, result in results.items():
            print(f"{model_name}:")
            print(f"  RMSE: {result['rmse']:.4f}")
            print(f"  MAE:  {result['mae']:.4f}")
        
        return results
    
    def generate_temporal_report(self, output_path=None):
        """
        生成时间序列分析报告
        
        Parameters:
        -----------
        output_path : str, optional
            报告保存路径
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        print("=" * 60)
        print("时间序列分析报告")
        print("=" * 60)
        
        # 基础统计
        print(f"\n1. 数据概览:")
        print(f"   数据时间范围: {self.df[self.time_column].min()} 到 {self.df[self.time_column].max()}")
        print(f"   数据点数量: {len(self.df)}")
        print(f"   数值范围: {self.df[self.value_column].min():.2f} - {self.df[self.value_column].max():.2f}")
        print(f"   平均值: {self.df[self.value_column].mean():.2f}")
        print(f"   标准差: {self.df[self.value_column].std():.2f}")
        
        # 时间模式分析
        print(f"\n2. 时间模式分析:")
        patterns = self.analyze_hourly_patterns('hour')
        peak_hour = patterns['hourly']['mean'].idxmax()
        print(f"   需求高峰时段: {peak_hour}:00")
        print(f"   高峰时段平均需求: {patterns['hourly']['mean'].max():.2f}")
        
        # 平稳性检验
        print(f"\n3. 平稳性检验:")
        stationarity = self.stationarity_test()
        
        # 模型预测
        print(f"\n4. 预测模型比较:")
        model_comparison = self.compare_models()
        
        print("\n" + "=" * 60)
        print("报告生成完成")
        print("=" * 60)


if __name__ == "__main__":
    # 示例用法
    print("时间序列分析模块")
    print("请导入并使用 TemporalAnalysis 类进行时间序列分析") 
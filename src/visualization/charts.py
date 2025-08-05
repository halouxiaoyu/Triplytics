"""
图表生成模块

该模块提供对滴滴出行数据的可视化功能，包括：
- 基础统计图表
- 时间序列图表
- 地理分布图表
- 交互式图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ChartGenerator:
    """图表生成器类"""
    
    def __init__(self, df=None):
        """
        初始化图表生成器
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            数据DataFrame
        """
        self.df = df
        self.color_palette = sns.color_palette("husl", 10)
        
    def set_data(self, df):
        """
        设置数据
        
        Parameters:
        -----------
        df : pandas.DataFrame
            数据DataFrame
        """
        self.df = df.copy()
        print(f"数据设置完成，样本数量: {len(self.df)}")
        
    def plot_time_series(self, time_column, value_column, figsize=(15, 8), 
                        title="时间序列图", interactive=True):
        """
        绘制时间序列图
        
        Parameters:
        -----------
        time_column : str
            时间列名
        value_column : str
            数值列名
        figsize : tuple
            图表大小
        title : str
            图表标题
        interactive : bool
            是否使用交互式图表
            
        Returns:
        --------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 确保时间列格式正确
        df_plot = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_plot[time_column]):
            df_plot[time_column] = pd.to_datetime(df_plot[time_column])
            
        df_plot = df_plot.sort_values(time_column)
        
        if interactive:
            # 使用Plotly绘制交互式图表
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_plot[time_column],
                y=df_plot[value_column],
                mode='lines+markers',
                name=value_column,
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="时间",
                yaxis_title=value_column,
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig.show()
            return fig
        else:
            # 使用Matplotlib绘制静态图表
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(df_plot[time_column], df_plot[value_column], 
                   linewidth=2, marker='o', markersize=4)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel(value_column, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            return fig
    
    def plot_hourly_pattern(self, time_column, value_column, figsize=(15, 10)):
        """
        绘制小时级模式图
        
        Parameters:
        -----------
        time_column : str
            时间列名
        value_column : str
            数值列名
        figsize : tuple
            图表大小
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        df_plot = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_plot[time_column]):
            df_plot[time_column] = pd.to_datetime(df_plot[time_column])
            
        # 提取时间特征
        df_plot['hour'] = df_plot[time_column].dt.hour
        df_plot['weekday'] = df_plot[time_column].dt.dayofweek
        df_plot['weekday_name'] = df_plot[time_column].dt.day_name()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 24小时模式
        hourly_avg = df_plot.groupby('hour')[value_column].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values, 
                       marker='o', linewidth=2, color='blue')
        axes[0, 0].set_title('24小时需求模式', fontweight='bold')
        axes[0, 0].set_xlabel('小时')
        axes[0, 0].set_ylabel(f'平均{value_column}')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(0, 24, 2))
        
        # 2. 工作日vs周末
        weekday_avg = df_plot.groupby('weekday')[value_column].mean()
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        bars = axes[0, 1].bar(weekday_names, weekday_avg.values, 
                             color=self.color_palette[:7])
        axes[0, 1].set_title('工作日需求模式', fontweight='bold')
        axes[0, 1].set_xlabel('星期')
        axes[0, 1].set_ylabel(f'平均{value_column}')
        
        # 添加数值标签
        for bar, value in zip(bars, weekday_avg.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # 3. 热力图：小时 x 星期
        pivot_table = df_plot.pivot_table(
            values=value_column, 
            index='weekday_name', 
            columns='hour', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=axes[1, 0], cbar_kws={'label': value_column})
        axes[1, 0].set_title('需求热力图 (星期 x 小时)', fontweight='bold')
        
        # 4. 日需求趋势
        daily_avg = df_plot.groupby(df_plot[time_column].dt.date)[value_column].mean()
        axes[1, 1].plot(daily_avg.index, daily_avg.values, linewidth=1, color='green')
        axes[1, 1].set_title('日需求趋势', fontweight='bold')
        axes[1, 1].set_xlabel('日期')
        axes[1, 1].set_ylabel(f'平均{value_column}')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 旋转x轴标签
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_distribution(self, column, bins=30, figsize=(12, 8), 
                         title=None, interactive=True):
        """
        绘制分布图
        
        Parameters:
        -----------
        column : str
            列名
        bins : int
            直方图箱数
        figsize : tuple
            图表大小
        title : str
            图表标题
        interactive : bool
            是否使用交互式图表
            
        Returns:
        --------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        if title is None:
            title = f"{column} 分布图"
            
        if interactive:
            # 使用Plotly绘制交互式图表
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=self.df[column],
                nbinsx=bins,
                name=column,
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=column,
                yaxis_title="频数",
                template='plotly_white',
                showlegend=False
            )
            
            fig.show()
            return fig
        else:
            # 使用Matplotlib绘制静态图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # 直方图
            ax1.hist(self.df[column], bins=bins, alpha=0.7, edgecolor='black', 
                    color='lightblue')
            ax1.set_title(f'{title} - 直方图', fontweight='bold')
            ax1.set_xlabel(column)
            ax1.set_ylabel('频数')
            ax1.grid(True, alpha=0.3)
            
            # 箱线图
            ax2.boxplot(self.df[column], patch_artist=True, 
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
            ax2.set_title(f'{title} - 箱线图', fontweight='bold')
            ax2.set_ylabel(column)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return fig
    
    def plot_correlation_matrix(self, columns=None, figsize=(12, 10), 
                               title="相关性矩阵"):
        """
        绘制相关性矩阵热力图
        
        Parameters:
        -----------
        columns : list, optional
            要分析的列名列表
        figsize : tuple
            图表大小
        title : str
            图表标题
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        if columns is None:
            # 选择数值列
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        # 计算相关性矩阵
        corr_matrix = self.df[columns].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建掩码，隐藏上三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 绘制热力图
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_scatter_matrix(self, columns=None, figsize=(15, 15)):
        """
        绘制散点图矩阵
        
        Parameters:
        -----------
        columns : list, optional
            要分析的列名列表
        figsize : tuple
            图表大小
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        if columns is None:
            # 选择数值列
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(columns) > 6:  # 限制列数避免图表过于复杂
                columns = columns[:6]
        
        fig = sns.pairplot(self.df[columns], diag_kind='kde', 
                          plot_kws={'alpha': 0.6, 's': 20})
        fig.fig.set_size_inches(figsize)
        fig.fig.suptitle('散点图矩阵', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
        
        return fig
    
    def plot_geographic_distribution(self, lat_column, lng_column, 
                                   value_column=None, figsize=(15, 10)):
        """
        绘制地理分布图
        
        Parameters:
        -----------
        lat_column : str
            纬度列名
        lng_column : str
            经度列名
        value_column : str, optional
            数值列名（用于颜色映射）
        figsize : tuple
            图表大小
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        df_plot = self.df.copy()
        
        # 移除异常坐标
        df_plot = df_plot[
            (df_plot[lat_column] >= -90) & (df_plot[lat_column] <= 90) &
            (df_plot[lng_column] >= -180) & (df_plot[lng_column] <= 180)
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. 散点图
        if value_column:
            scatter = ax1.scatter(df_plot[lng_column], df_plot[lat_column], 
                                c=df_plot[value_column], cmap='viridis', 
                                alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax1, label=value_column)
            ax1.set_title(f'地理分布图 - {value_column}', fontweight='bold')
        else:
            ax1.scatter(df_plot[lng_column], df_plot[lat_column], 
                       alpha=0.6, s=20, color='blue')
            ax1.set_title('地理分布图', fontweight='bold')
            
        ax1.set_xlabel('经度')
        ax1.set_ylabel('纬度')
        ax1.grid(True, alpha=0.3)
        
        # 2. 密度图
        if value_column:
            # 创建网格
            x_min, x_max = df_plot[lng_column].min(), df_plot[lng_column].max()
            y_min, y_max = df_plot[lat_column].min(), df_plot[lat_column].max()
            
            xi = np.linspace(x_min, x_max, 100)
            yi = np.linspace(y_min, y_max, 100)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # 计算密度
            from scipy.stats import gaussian_kde
            positions = np.vstack([xi_grid.ravel(), yi_grid.ravel()])
            values = np.vstack([df_plot[lng_column], df_plot[lat_column]])
            kernel = gaussian_kde(values)
            zi = np.reshape(kernel(positions).T, xi_grid.shape)
            
            contour = ax2.contourf(xi_grid, yi_grid, zi, levels=20, cmap='viridis')
            plt.colorbar(contour, ax=ax2, label='密度')
            ax2.set_title('密度分布图', fontweight='bold')
        else:
            ax2.hexbin(df_plot[lng_column], df_plot[lat_column], 
                      gridsize=50, cmap='viridis')
            ax2.set_title('密度分布图', fontweight='bold')
            
        ax2.set_xlabel('经度')
        ax2.set_ylabel('纬度')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_interactive_dashboard(self, time_column, value_column, 
                                 lat_column=None, lng_column=None):
        """
        创建交互式仪表板
        
        Parameters:
        -----------
        time_column : str
            时间列名
        value_column : str
            数值列名
        lat_column : str, optional
            纬度列名
        lng_column : str, optional
            经度列名
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        df_plot = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_plot[time_column]):
            df_plot[time_column] = pd.to_datetime(df_plot[time_column])
            
        # 提取时间特征
        df_plot['hour'] = df_plot[time_column].dt.hour
        df_plot['weekday'] = df_plot[time_column].dt.dayofweek
        df_plot['weekday_name'] = df_plot[time_column].dt.day_name()
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('时间序列', '小时模式', '工作日模式', '地理分布'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "scattergeo"}]]
        )
        
        # 1. 时间序列
        fig.add_trace(
            go.Scatter(x=df_plot[time_column], y=df_plot[value_column],
                      mode='lines', name='时间序列'),
            row=1, col=1
        )
        
        # 2. 小时模式
        hourly_avg = df_plot.groupby('hour')[value_column].mean()
        fig.add_trace(
            go.Bar(x=hourly_avg.index, y=hourly_avg.values,
                  name='小时模式'),
            row=1, col=2
        )
        
        # 3. 工作日模式
        weekday_avg = df_plot.groupby('weekday')[value_column].mean()
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        fig.add_trace(
            go.Bar(x=weekday_names, y=weekday_avg.values,
                  name='工作日模式'),
            row=2, col=1
        )
        
        # 4. 地理分布（如果有坐标）
        if lat_column and lng_column:
            fig.add_trace(
                go.Scattergeo(
                    lon=df_plot[lng_column],
                    lat=df_plot[lat_column],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=df_plot[value_column],
                        colorscale='Viridis',
                        opacity=0.7,
                        colorbar=dict(title=value_column)
                    ),
                    name='地理分布'
                ),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="滴滴出行数据分析仪表板",
            showlegend=False,
            height=800
        )
        
        # 更新x轴标签
        fig.update_xaxes(title_text="时间", row=1, col=1)
        fig.update_xaxes(title_text="小时", row=1, col=2)
        fig.update_xaxes(title_text="星期", row=2, col=1)
        
        # 更新y轴标签
        fig.update_yaxes(title_text=value_column, row=1, col=1)
        fig.update_yaxes(title_text=f"平均{value_column}", row=1, col=2)
        fig.update_yaxes(title_text=f"平均{value_column}", row=2, col=1)
        
        fig.show()
        return fig
    
    def plot_animated_chart(self, time_column, value_column, 
                          animation_column=None, figsize=(15, 8)):
        """
        绘制动画图表
        
        Parameters:
        -----------
        time_column : str
            时间列名
        value_column : str
            数值列名
        animation_column : str, optional
            动画帧列名
        figsize : tuple
            图表大小
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        df_plot = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_plot[time_column]):
            df_plot[time_column] = pd.to_datetime(df_plot[time_column])
            
        if animation_column is None:
            # 使用小时作为动画帧
            df_plot['hour'] = df_plot[time_column].dt.hour
            animation_column = 'hour'
        
        # 创建动画散点图
        fig = px.scatter(
            df_plot,
            x=time_column,
            y=value_column,
            animation_frame=animation_column,
            title=f"{value_column} 随时间变化动画",
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title=value_column
        )
        
        fig.show()
        return fig
    
    def generate_summary_charts(self, value_columns=None, figsize=(20, 15)):
        """
        生成数据概览图表
        
        Parameters:
        -----------
        value_columns : list, optional
            要分析的数值列列表
        figsize : tuple
            图表大小
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        if value_columns is None:
            value_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(value_columns) > 6:
                value_columns = value_columns[:6]
        
        n_cols = len(value_columns)
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(value_columns):
            # 直方图
            axes[0, i].hist(self.df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f'{col} 分布', fontweight='bold')
            axes[0, i].set_xlabel(col)
            axes[0, i].set_ylabel('频数')
            axes[0, i].grid(True, alpha=0.3)
            
            # 箱线图
            axes[1, i].boxplot(self.df[col], patch_artist=True)
            axes[1, i].set_title(f'{col} 箱线图', fontweight='bold')
            axes[1, i].set_ylabel(col)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


if __name__ == "__main__":
    # 示例用法
    print("图表生成模块")
    print("请导入并使用 ChartGenerator 类进行数据可视化") 
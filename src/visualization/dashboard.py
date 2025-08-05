"""
交互式仪表盘模块
基于Streamlit构建的可视化仪表盘
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import altair as alt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data.data_loader import DataLoader
from src.analysis.eda import ExploratoryDataAnalysis
from src.analysis.spatial_analysis import SpatialAnalysis
from src.analysis.temporal_analysis import TemporalAnalysis
from src.models.prediction import DemandPredictor
from src.visualization.charts import ChartGenerator
from src.visualization.maps import MapVisualizer

class InteractiveDashboard:
    """交互式仪表盘类"""
    
    def __init__(self):
        """初始化仪表盘"""
        self.data_loader = DataLoader()
        self.chart_gen = ChartGenerator()
        self.map_viz = MapVisualizer()
        
    def setup_page(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="滴滴出行数据分析仪表盘",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 自定义CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def load_data(self):
        """加载数据"""
        try:
            # 尝试加载处理后的数据
            df = self.data_loader.get_latest_processed_data()
            if df is None or len(df) == 0:
                # 如果没有处理后的数据，加载原始数据
                df = self.data_loader.load_uber_data()
                df = df.sample(min(10000, len(df)), random_state=42)
            return df
        except Exception as e:
            st.error(f"数据加载失败: {str(e)}")
            # 尝试加载原始数据作为备选
            try:
                df = self.data_loader.load_uber_data()
                df = df.sample(min(10000, len(df)), random_state=42)
                st.warning("使用原始数据，建议先运行数据处理流程")
                return df
            except Exception as e2:
                st.error(f"原始数据加载也失败: {str(e2)}")
                return None
    
    def create_sidebar_filters(self, df):
        """创建侧边栏筛选器"""
        st.sidebar.header("📊 数据筛选")
        
        # 时间筛选
        if 'datetime' in df.columns:
            date_range = st.sidebar.date_input(
                "选择日期范围",
                value=[df['datetime'].min().date(), df['datetime'].max().date()],
                min_value=df['datetime'].min().date(),
                max_value=df['datetime'].max().date()
            )
        
        # 小时筛选
        if 'hour' in df.columns:
            hours = st.sidebar.multiselect(
                "选择小时",
                options=sorted(df['hour'].unique()),
                default=sorted(df['hour'].unique())[:6]
            )
        
        # 区域筛选
        if 'pickup_area' in df.columns:
            areas = st.sidebar.multiselect(
                "选择区域",
                options=['全部'] + sorted(df['pickup_area'].unique().tolist()),
                default=['全部']
            )
        
        # 应用筛选
        filtered_df = df.copy()
        
        if 'datetime' in df.columns and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['datetime'].dt.date >= date_range[0]) &
                (filtered_df['datetime'].dt.date <= date_range[1])
            ]
        
        if 'hour' in df.columns and hours:
            filtered_df = filtered_df[filtered_df['hour'].isin(hours)]
        
        if 'pickup_area' in df.columns and areas and '全部' not in areas:
            filtered_df = filtered_df[filtered_df['pickup_area'].isin(areas)]
        
        return filtered_df
    
    def display_key_metrics(self, df):
        """显示关键指标"""
        st.markdown('<h1 class="main-header">🚗 滴滴出行数据分析仪表盘</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "总订单数",
                f"{len(df):,}",
                delta=f"{len(df) - len(df) * 0.95:.0f}"
            )
        
        with col2:
            if 'trip_distance' in df.columns:
                avg_distance = df['trip_distance'].mean()
                st.metric(
                    "平均行程距离",
                    f"{avg_distance:.2f} km",
                    delta=f"{avg_distance * 0.05:.2f}"
                )
            else:
                st.metric("平均行程距离", "N/A")
        
        with col3:
            if 'is_peak_hour' in df.columns:
                peak_ratio = df['is_peak_hour'].mean() * 100
                st.metric(
                    "高峰时段占比",
                    f"{peak_ratio:.1f}%",
                    delta=f"{peak_ratio * 0.1:.1f}%"
                )
            else:
                st.metric("高峰时段占比", "N/A")
        
        with col4:
            if 'is_weekend' in df.columns:
                weekend_ratio = df['is_weekend'].mean() * 100
                st.metric(
                    "周末订单占比",
                    f"{weekend_ratio:.1f}%",
                    delta=f"{weekend_ratio * 0.05:.1f}%"
                )
            else:
                st.metric("周末订单占比", "N/A")
    
    def display_time_analysis(self, df):
        """显示时间分析"""
        st.header("⏰ 时间模式分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 小时分布
            if 'hour' in df.columns:
                hourly_counts = df['hour'].value_counts().sort_index()
                fig_hourly = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="24小时订单分布",
                    labels={'x': '小时', 'y': '订单数量'}
                )
                fig_hourly.update_layout(showlegend=False)
                st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # 星期分布
            if 'weekday' in df.columns:
                weekday_counts = df['weekday'].value_counts().sort_index()
                weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
                fig_weekday = px.bar(
                    x=[weekday_names[i] for i in weekday_counts.index],
                    y=weekday_counts.values,
                    title="星期订单分布",
                    labels={'x': '星期', 'y': '订单数量'}
                )
                fig_weekday.update_layout(showlegend=False)
                st.plotly_chart(fig_weekday, use_container_width=True)
        
        # 时间热力图
        if 'hour' in df.columns and 'weekday' in df.columns:
            st.subheader("时间热力图")
            time_heatmap = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
            
            fig_heatmap = px.imshow(
                time_heatmap,
                title="星期-小时热力图",
                labels=dict(x="小时", y="星期", color="订单数量"),
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def display_spatial_analysis(self, df):
        """显示空间分析"""
        st.header("🗺️ 空间分布分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 区域分布
            if 'pickup_area' in df.columns:
                area_counts = df['pickup_area'].value_counts().head(10)
                fig_area = px.bar(
                    x=area_counts.values,
                    y=area_counts.index,
                    orientation='h',
                    title="热门上车区域TOP10",
                    labels={'x': '订单数量', 'y': '区域'}
                )
                st.plotly_chart(fig_area, use_container_width=True)
        
        with col2:
            # 行程距离分布
            if 'trip_distance' in df.columns:
                fig_distance = px.histogram(
                    df,
                    x='trip_distance',
                    nbins=30,
                    title="行程距离分布",
                    labels={'trip_distance': '距离 (km)', 'count': '订单数量'}
                )
                st.plotly_chart(fig_distance, use_container_width=True)
        
        # 地图可视化
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.subheader("地理分布图")
            
            # 创建地图
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # 添加热力图
            heat_data = df[['latitude', 'longitude']].values.tolist()
            folium.plugins.HeatMap(heat_data).add_to(m)
            
            folium_static(m, width=800, height=400)
    
    def display_business_insights(self, df):
        """显示业务洞察"""
        st.header("💡 业务洞察")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 需求趋势
            if 'datetime' in df.columns:
                daily_counts = df.groupby(df['datetime'].dt.date).size()
                fig_trend = px.line(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    title="日订单趋势",
                    labels={'x': '日期', 'y': '订单数量'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # 高峰时段分析 - 完全重写以避免缓存问题
            if 'is_peak_hour' in df.columns:
                try:
                    peak_analysis = df.groupby('is_peak_hour').size()
                    
                    # 安全处理饼图数据
                    if len(peak_analysis) == 2:
                        # 有两种类型：高峰和非高峰
                        peak_data = {
                            '类型': ['非高峰', '高峰'],
                            '数量': peak_analysis.values
                        }
                        peak_df = pd.DataFrame(peak_data)
                        fig_peak = px.pie(
                            peak_df, 
                            values='数量', 
                            names='类型',
                            title="高峰时段分布"
                        )
                        st.plotly_chart(fig_peak, use_container_width=True)
                    elif len(peak_analysis) == 1:
                        # 只有一种类型
                        peak_type = "高峰" if peak_analysis.index[0] else "非高峰"
                        st.info(f"数据中只有{peak_type}时段的数据")
                    else:
                        # 有多种类型，使用实际索引
                        peak_data = {
                            '类型': peak_analysis.index.astype(str),
                            '数量': peak_analysis.values
                        }
                        peak_df = pd.DataFrame(peak_data)
                        fig_peak = px.pie(
                            peak_df, 
                            values='数量', 
                            names='类型',
                            title="时段分布"
                        )
                        st.plotly_chart(fig_peak, use_container_width=True)
                except Exception as e:
                    st.error(f"高峰时段分析出错: {str(e)}")
                    st.info("跳过高峰时段分析")
        
        # 相关性分析
        st.subheader("特征相关性分析")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # 限制列数
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="特征相关性热力图",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    def display_prediction_interface(self, df):
        """显示预测接口"""
        st.header("🔮 需求预测")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("预测参数设置")
            
            # 预测时间
            pred_date = st.date_input("预测日期", value=datetime.now().date())
            pred_hour = st.selectbox("预测小时", options=list(range(24)))
            
            # 预测区域
            if 'pickup_area' in df.columns:
                pred_area = st.selectbox("预测区域", options=df['pickup_area'].unique())
            else:
                pred_area = "默认区域"
            
            # 预测按钮
            if st.button("开始预测", type="primary"):
                with st.spinner("正在预测..."):
                    # 这里可以调用预测模型
                    prediction = self.run_prediction(df, pred_date, pred_hour, pred_area)
                    st.success(f"预测完成！预计需求: {prediction:.0f} 单")
        
        with col2:
            st.subheader("预测结果可视化")
            
            # 模拟预测结果
            hours = list(range(24))
            predictions = np.random.poisson(50, 24)  # 模拟预测值
            
            fig_pred = px.line(
                x=hours,
                y=predictions,
                title="24小时需求预测",
                labels={'x': '小时', 'y': '预测需求'}
            )
            st.plotly_chart(fig_pred, use_container_width=True)
    
    def run_prediction(self, df, date, hour, area):
        """运行预测模型"""
        # 这里可以集成实际的预测模型
        # 目前返回模拟值
        base_demand = 50
        hour_factor = 1.5 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0.8
        area_factor = 1.2 if area in df['pickup_area'].value_counts().head(5).index else 0.9
        
        return base_demand * hour_factor * area_factor
    
    def display_real_time_monitoring(self):
        """显示实时监控"""
        st.header("📡 实时监控")
        
        # 模拟实时数据
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("当前在线车辆", "1,234", delta="+12")
        
        with col2:
            st.metric("当前等待订单", "89", delta="-5")
        
        with col3:
            st.metric("平均等待时间", "3.2分钟", delta="-0.5")
        
        # 实时图表
        st.subheader("实时订单趋势")
        
        # 模拟实时数据
        time_points = pd.date_range(start=datetime.now() - timedelta(hours=6), 
                                  end=datetime.now(), freq='10min')
        orders = np.random.poisson(20, len(time_points))
        
        fig_realtime = px.line(
            x=time_points,
            y=orders,
            title="最近6小时订单趋势",
            labels={'x': '时间', 'y': '订单数量'}
        )
        st.plotly_chart(fig_realtime, use_container_width=True)
        
        # 自动刷新
        if st.button("刷新数据"):
            st.rerun()
    
    def run_dashboard(self):
        """运行仪表盘"""
        self.setup_page()
        
        # 加载数据
        df = self.load_data()
        if df is None:
            st.error("无法加载数据，请检查数据文件")
            return
        
        # 创建筛选器
        filtered_df = self.create_sidebar_filters(df)
        
        # 显示关键指标
        self.display_key_metrics(filtered_df)
        
        # 创建标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 时间分析", "🗺️ 空间分析", "💡 业务洞察", 
            "🔮 需求预测", "📡 实时监控"
        ])
        
        with tab1:
            self.display_time_analysis(filtered_df)
        
        with tab2:
            self.display_spatial_analysis(filtered_df)
        
        with tab3:
            self.display_business_insights(filtered_df)
        
        with tab4:
            self.display_prediction_interface(filtered_df)
        
        with tab5:
            
            self.display_real_time_monitoring()

def main():
    """主函数"""
    dashboard = InteractiveDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 
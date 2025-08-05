"""
实时分析模块
支持流数据处理、实时监控和动态预测
"""

import pandas as pd
import numpy as np
import time
import threading
import queue
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from collections import deque, defaultdict
from typing import Dict, List, Optional, Callable, Any
import json

from config.logging_config import LoggerMixin, log_execution_time
from src.models.prediction import DemandPredictor
from src.analysis.temporal_analysis import TemporalAnalyzer
from src.analysis.spatial_analysis import SpatialAnalyzer

class RealTimeAnalyzer(LoggerMixin):
    """实时分析器类"""
    
    def __init__(self, window_size: int = 1000, update_interval: int = 60):
        """
        初始化实时分析器
        
        Args:
            window_size: 滑动窗口大小
            update_interval: 更新间隔（秒）
        """
        super().__init__()
        self.window_size = window_size
        self.update_interval = update_interval
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=window_size)
        self.time_series_buffer = deque(maxlen=window_size)
        
        # 实时统计
        self.real_time_stats = {
            'total_orders': 0,
            'current_hour_orders': 0,
            'peak_hour_orders': 0,
            'avg_wait_time': 0.0,
            'active_vehicles': 0,
            'demand_surge': False
        }
        
        # 预测器
        self.demand_predictor = DemandPredictor()
        self.temporal_analyzer = TemporalAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        
        # 回调函数
        self.callbacks = {
            'demand_surge': [],
            'anomaly_detected': [],
            'prediction_update': [],
            'stats_update': []
        }
        
        # 线程控制
        self.is_running = False
        self.analysis_thread = None
        self.data_queue = queue.Queue()
        
    def start_realtime_analysis(self):
        """启动实时分析"""
        if self.is_running:
            self.log_warning("实时分析已在运行")
            return
        
        self.is_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        self.log_info("实时分析已启动")
    
    def stop_realtime_analysis(self):
        """停止实时分析"""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join()
        self.log_info("实时分析已停止")
    
    def add_data_point(self, data_point: Dict[str, Any]):
        """
        添加数据点
        
        Args:
            data_point: 数据点字典
        """
        try:
            # 添加到队列
            self.data_queue.put(data_point)
            
            # 添加到缓冲区
            self.data_buffer.append(data_point)
            
            # 更新时间序列
            if 'timestamp' in data_point:
                self.time_series_buffer.append(data_point['timestamp'])
            
        except Exception as e:
            self.log_error(f"添加数据点失败: {str(e)}")
    
    def _analysis_loop(self):
        """分析循环"""
        while self.is_running:
            try:
                # 处理队列中的数据
                while not self.data_queue.empty():
                    data_point = self.data_queue.get_nowait()
                    self._process_data_point(data_point)
                
                # 定期更新统计和预测
                self._update_statistics()
                self._update_predictions()
                self._check_anomalies()
                
                # 触发回调
                self._trigger_callbacks()
                
                # 等待下次更新
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.log_error(f"分析循环错误: {str(e)}")
                time.sleep(5)  # 错误后等待5秒
    
    def _process_data_point(self, data_point: Dict[str, Any]):
        """处理单个数据点"""
        # 更新基础统计
        self.real_time_stats['total_orders'] += 1
        
        # 检查是否当前小时
        current_hour = datetime.now().hour
        if 'timestamp' in data_point:
            data_hour = pd.to_datetime(data_point['timestamp']).hour
            if data_hour == current_hour:
                self.real_time_stats['current_hour_orders'] += 1
        
        # 检查高峰时段
        if self._is_peak_hour(current_hour):
            self.real_time_stats['peak_hour_orders'] += 1
        
        # 更新等待时间
        if 'wait_time' in data_point:
            self._update_avg_wait_time(data_point['wait_time'])
        
        # 更新活跃车辆数
        if 'vehicle_id' in data_point:
            self._update_active_vehicles(data_point['vehicle_id'])
    
    def _update_statistics(self):
        """更新实时统计"""
        if len(self.data_buffer) == 0:
            return
        
        # 计算需求激增
        recent_orders = len([d for d in self.data_buffer 
                           if pd.to_datetime(d.get('timestamp', datetime.now())) > 
                           datetime.now() - timedelta(minutes=30)])
        
        avg_orders_per_minute = recent_orders / 30
        self.real_time_stats['demand_surge'] = avg_orders_per_minute > 2.0  # 阈值可调
        
        # 更新其他统计
        self._calculate_spatial_stats()
        self._calculate_temporal_stats()
    
    def _update_predictions(self):
        """更新预测"""
        if len(self.data_buffer) < 10:  # 需要足够的数据
            return
        
        try:
            # 准备预测数据
            recent_data = list(self.data_buffer)[-100:]  # 最近100个数据点
            df_recent = pd.DataFrame(recent_data)
            
            # 生成短期预测
            prediction = self._generate_short_term_prediction(df_recent)
            
            # 触发预测更新回调
            self._trigger_callback('prediction_update', prediction)
            
        except Exception as e:
            self.log_error(f"预测更新失败: {str(e)}")
    
    def _check_anomalies(self):
        """检查异常"""
        if len(self.data_buffer) < 20:
            return
        
        try:
            # 检测异常模式
            anomalies = self._detect_anomalies()
            
            if anomalies:
                self._trigger_callback('anomaly_detected', anomalies)
                
        except Exception as e:
            self.log_error(f"异常检测失败: {str(e)}")
    
    def _generate_short_term_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成短期预测"""
        try:
            # 简单的移动平均预测
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # 按分钟聚合
                df_minute = df.groupby(df['timestamp'].dt.floor('min')).size().reset_index()
                df_minute.columns = ['timestamp', 'orders']
                
                # 计算移动平均
                if len(df_minute) >= 5:
                    moving_avg = df_minute['orders'].rolling(window=5, min_periods=1).mean()
                    
                    # 预测未来5分钟
                    future_predictions = []
                    for i in range(5):
                        if i == 0:
                            pred = moving_avg.iloc[-1]
                        else:
                            pred = moving_avg.iloc[-1] * (1 + 0.1 * i)  # 简单趋势
                        future_predictions.append(max(0, pred))
                    
                    return {
                        'timestamp': datetime.now(),
                        'predictions': future_predictions,
                        'confidence': 0.8,
                        'method': 'moving_average'
                    }
            
            return {
                'timestamp': datetime.now(),
                'predictions': [0] * 5,
                'confidence': 0.0,
                'method': 'insufficient_data'
            }
            
        except Exception as e:
            self.log_error(f"短期预测生成失败: {str(e)}")
            return None
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        try:
            # 检测订单量异常
            recent_orders = len([d for d in self.data_buffer 
                               if pd.to_datetime(d.get('timestamp', datetime.now())) > 
                               datetime.now() - timedelta(minutes=10)])
            
            # 如果10分钟内订单量超过阈值，标记为异常
            if recent_orders > 50:  # 阈值可调
                anomalies.append({
                    'type': 'high_demand',
                    'timestamp': datetime.now(),
                    'value': recent_orders,
                    'threshold': 50,
                    'severity': 'high' if recent_orders > 100 else 'medium'
                })
            
            # 检测等待时间异常
            if self.real_time_stats['avg_wait_time'] > 10:  # 超过10分钟
                anomalies.append({
                    'type': 'high_wait_time',
                    'timestamp': datetime.now(),
                    'value': self.real_time_stats['avg_wait_time'],
                    'threshold': 10,
                    'severity': 'high'
                })
            
            # 检测需求激增
            if self.real_time_stats['demand_surge']:
                anomalies.append({
                    'type': 'demand_surge',
                    'timestamp': datetime.now(),
                    'severity': 'high'
                })
                
        except Exception as e:
            self.log_error(f"异常检测失败: {str(e)}")
        
        return anomalies
    
    def _calculate_spatial_stats(self):
        """计算空间统计"""
        if len(self.data_buffer) == 0:
            return
        
        try:
            # 统计热门区域
            locations = [d.get('pickup_area', 'unknown') for d in self.data_buffer 
                        if 'pickup_area' in d]
            
            if locations:
                location_counts = pd.Series(locations).value_counts()
                self.real_time_stats['hotspots'] = location_counts.head(5).to_dict()
            
        except Exception as e:
            self.log_error(f"空间统计计算失败: {str(e)}")
    
    def _calculate_temporal_stats(self):
        """计算时间统计"""
        if len(self.time_series_buffer) == 0:
            return
        
        try:
            # 计算当前小时订单量
            current_hour = datetime.now().hour
            hour_orders = len([t for t in self.time_series_buffer 
                             if pd.to_datetime(t).hour == current_hour])
            
            self.real_time_stats['current_hour_orders'] = hour_orders
            
        except Exception as e:
            self.log_error(f"时间统计计算失败: {str(e)}")
    
    def _update_avg_wait_time(self, wait_time: float):
        """更新平均等待时间"""
        if wait_time <= 0:
            return
        
        # 简单的指数移动平均
        alpha = 0.1
        current_avg = self.real_time_stats['avg_wait_time']
        
        if current_avg == 0:
            self.real_time_stats['avg_wait_time'] = wait_time
        else:
            self.real_time_stats['avg_wait_time'] = alpha * wait_time + (1 - alpha) * current_avg
    
    def _update_active_vehicles(self, vehicle_id: str):
        """更新活跃车辆数"""
        # 这里可以实现更复杂的车辆跟踪逻辑
        # 目前只是简单的计数
        pass
    
    def _is_peak_hour(self, hour: int) -> bool:
        """判断是否为高峰时段"""
        return (7 <= hour <= 9) or (17 <= hour <= 19)
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        添加回调函数
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            self.log_info(f"已添加 {event_type} 回调函数")
        else:
            self.log_warning(f"未知事件类型: {event_type}")
    
    def _trigger_callback(self, event_type: str, data: Any = None):
        """触发回调函数"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.log_error(f"回调函数执行失败: {str(e)}")
    
    def _trigger_callbacks(self):
        """触发统计更新回调"""
        self._trigger_callback('stats_update', self.real_time_stats)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计"""
        return self.real_time_stats.copy()
    
    def get_recent_data(self, minutes: int = 60) -> pd.DataFrame:
        """
        获取最近的数据
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            最近的数据DataFrame
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_data = []
        for data_point in self.data_buffer:
            if 'timestamp' in data_point:
                timestamp = pd.to_datetime(data_point['timestamp'])
                if timestamp > cutoff_time:
                    recent_data.append(data_point)
        
        return pd.DataFrame(recent_data)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'buffer_size': len(self.data_buffer),
            'queue_size': self.data_queue.qsize(),
            'is_running': self.is_running,
            'update_interval': self.update_interval,
            'last_update': datetime.now()
        }

class StreamDataProcessor:
    """流数据处理器"""
    
    def __init__(self, batch_size: int = 100):
        """
        初始化流数据处理器
        
        Args:
            batch_size: 批处理大小
        """
        self.batch_size = batch_size
        self.data_buffer = []
        self.processors = []
        
    def add_processor(self, processor: Callable):
        """添加数据处理器"""
        self.processors.append(processor)
    
    def process_data(self, data: Dict[str, Any]):
        """处理数据"""
        self.data_buffer.append(data)
        
        # 达到批处理大小时处理
        if len(self.data_buffer) >= self.batch_size:
            self._process_batch()
    
    def _process_batch(self):
        """处理批次数据"""
        if not self.data_buffer:
            return
        
        batch_data = self.data_buffer.copy()
        self.data_buffer.clear()
        
        # 应用所有处理器
        for processor in self.processors:
            try:
                batch_data = processor(batch_data)
            except Exception as e:
                print(f"处理器执行失败: {str(e)}")
        
        return batch_data

class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        """
        初始化实时监控器
        
        Args:
            alert_thresholds: 告警阈值
        """
        self.alert_thresholds = alert_thresholds or {
            'high_demand': 50,
            'high_wait_time': 10,
            'low_vehicle_availability': 0.3
        }
        
        self.alerts = []
        self.monitoring_rules = []
        
    def add_monitoring_rule(self, rule: Callable):
        """添加监控规则"""
        self.monitoring_rules.append(rule)
    
    def check_alerts(self, current_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警"""
        new_alerts = []
        
        # 检查阈值告警
        for metric, threshold in self.alert_thresholds.items():
            if metric in current_stats:
                value = current_stats[metric]
                if isinstance(value, (int, float)) and value > threshold:
                    new_alerts.append({
                        'type': 'threshold_exceeded',
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'timestamp': datetime.now(),
                        'severity': 'high' if value > threshold * 1.5 else 'medium'
                    })
        
        # 应用自定义规则
        for rule in self.monitoring_rules:
            try:
                rule_alerts = rule(current_stats)
                if rule_alerts:
                    new_alerts.extend(rule_alerts)
            except Exception as e:
                print(f"监控规则执行失败: {str(e)}")
        
        # 更新告警列表
        self.alerts.extend(new_alerts)
        
        # 清理旧告警（保留最近24小时）
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts 
                      if alert['timestamp'] > cutoff_time]
        
        return new_alerts
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return self.alerts.copy()

def create_realtime_analyzer(window_size: int = 1000, update_interval: int = 60):
    """创建实时分析器"""
    return RealTimeAnalyzer(window_size, update_interval)

def create_stream_processor(batch_size: int = 100):
    """创建流数据处理器"""
    return StreamDataProcessor(batch_size)

def create_realtime_monitor(alert_thresholds: Dict[str, float] = None):
    """创建实时监控器"""
    return RealTimeMonitor(alert_thresholds)

if __name__ == "__main__":
    # 示例用法
    analyzer = RealTimeAnalyzer()
    print("实时分析模块已加载") 
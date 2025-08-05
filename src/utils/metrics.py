"""
评估指标模块

该模块提供各种评估指标的计算功能，包括：
- 回归指标
- 分类指标
- 时间序列指标
- 自定义指标
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """评估指标计算器类"""
    
    def __init__(self):
        """初始化评估指标计算器"""
        pass
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """
        计算回归指标
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
            
        Returns:
        --------
        dict : 回归指标
        """
        # 基础指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 相对误差指标
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 调整R²
        n = len(y_true)
        p = 1  # 假设特征数量为1，实际使用时需要传入
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # 平均绝对百分比误差
        mape_robust = np.median(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 对称平均绝对百分比误差
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'mape': mape,
            'mape_robust': mape_robust,
            'smape': smape
        }
    
    def calculate_classification_metrics(self, y_true, y_pred, y_prob=None):
        """
        计算分类指标
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        y_prob : array-like, optional
            预测概率
            
        Returns:
        --------
        dict : 分类指标
        """
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算每个类别的指标
        class_precision = precision_score(y_true, y_pred, average=None)
        class_recall = recall_score(y_true, y_pred, average=None)
        class_f1 = f1_score(y_true, y_pred, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'class_precision': class_precision,
            'class_recall': class_recall,
            'class_f1': class_f1
        }
        
        # 如果有概率预测，计算ROC AUC
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) > 2:
                    # 多分类ROC AUC
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                else:
                    # 二分类ROC AUC
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
                metrics['roc_auc'] = roc_auc
            except Exception as e:
                print(f"计算ROC AUC时出错: {e}")
                metrics['roc_auc'] = None
        
        return metrics
    
    def calculate_time_series_metrics(self, y_true, y_pred):
        """
        计算时间序列指标
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
            
        Returns:
        --------
        dict : 时间序列指标
        """
        # 基础回归指标
        regression_metrics = self.calculate_regression_metrics(y_true, y_pred)
        
        # 时间序列特定指标
        residuals = y_true - y_pred
        
        # 平均绝对误差
        mae = np.mean(np.abs(residuals))
        
        # 平均绝对百分比误差
        mape = np.mean(np.abs(residuals / y_true)) * 100
        
        # 均方根误差
        rmse = np.sqrt(np.mean(residuals**2))
        
        # 平均绝对偏差
        mad = np.median(np.abs(residuals - np.median(residuals)))
        
        # 方向准确率（预测趋势的准确性）
        direction_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
        
        # 趋势一致性
        trend_consistency = np.corrcoef(np.diff(y_true), np.diff(y_pred))[0, 1]
        
        time_series_metrics = {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'mad': mad,
            'direction_accuracy': direction_accuracy,
            'trend_consistency': trend_consistency
        }
        
        # 合并指标
        all_metrics = {**regression_metrics, **time_series_metrics}
        
        return all_metrics
    
    def calculate_custom_metrics(self, y_true, y_pred, metric_type='business'):
        """
        计算自定义业务指标
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        metric_type : str
            指标类型：'business', 'financial', 'operational'
            
        Returns:
        --------
        dict : 自定义指标
        """
        if metric_type == 'business':
            # 业务指标
            overestimation_penalty = np.mean(np.where(y_pred > y_true, 
                                                    (y_pred - y_true) / y_true, 0)) * 100
            underestimation_penalty = np.mean(np.where(y_pred < y_true, 
                                                     (y_true - y_pred) / y_true, 0)) * 100
            
            # 预测偏差
            bias = np.mean(y_pred - y_true)
            
            # 预测稳定性（预测值的变化程度）
            prediction_stability = np.std(y_pred)
            
            return {
                'overestimation_penalty': overestimation_penalty,
                'underestimation_penalty': underestimation_penalty,
                'bias': bias,
                'prediction_stability': prediction_stability
            }
        
        elif metric_type == 'financial':
            # 财务指标
            revenue_loss = np.sum(np.maximum(0, y_true - y_pred))  # 低估造成的收入损失
            cost_overrun = np.sum(np.maximum(0, y_pred - y_true))  # 高估造成的成本超支
            
            return {
                'revenue_loss': revenue_loss,
                'cost_overrun': cost_overrun,
                'total_financial_impact': revenue_loss + cost_overrun
            }
        
        elif metric_type == 'operational':
            # 运营指标
            # 预测准确度（在可接受误差范围内的预测比例）
            acceptable_error = 0.1  # 10%的可接受误差
            accuracy_within_tolerance = np.mean(np.abs((y_pred - y_true) / y_true) <= acceptable_error)
            
            # 预测一致性（连续预测的稳定性）
            prediction_consistency = 1 - np.std(np.diff(y_pred)) / np.mean(y_pred)
            
            return {
                'accuracy_within_tolerance': accuracy_within_tolerance,
                'prediction_consistency': prediction_consistency
            }
        
        else:
            raise ValueError(f"不支持的指标类型: {metric_type}")
    
    def calculate_model_comparison_metrics(self, models_results):
        """
        计算模型比较指标
        
        Parameters:
        -----------
        models_results : dict
            各模型的结果字典
            
        Returns:
        --------
        pandas.DataFrame : 模型比较结果
        """
        comparison_data = []
        
        for model_name, results in models_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                row = {'Model': model_name}
                row.update(metrics)
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 计算排名
        numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Model':
                # 对于某些指标，越小越好
                if col in ['mse', 'rmse', 'mae', 'mape']:
                    comparison_df[f'{col}_rank'] = comparison_df[col].rank()
                else:
                    comparison_df[f'{col}_rank'] = comparison_df[col].rank(ascending=False)
        
        return comparison_df
    
    def calculate_feature_importance_metrics(self, feature_importance_df):
        """
        计算特征重要性指标
        
        Parameters:
        -----------
        feature_importance_df : pandas.DataFrame
            特征重要性DataFrame
            
        Returns:
        --------
        dict : 特征重要性指标
        """
        if feature_importance_df is None or len(feature_importance_df) == 0:
            return {}
        
        # 基础统计
        total_importance = feature_importance_df['importance'].sum()
        mean_importance = feature_importance_df['importance'].mean()
        std_importance = feature_importance_df['importance'].std()
        
        # 重要性分布
        top_5_importance = feature_importance_df.head(5)['importance'].sum()
        top_10_importance = feature_importance_df.head(10)['importance'].sum()
        
        # 重要性集中度（前5个特征的重要性占比）
        concentration_ratio = top_5_importance / total_importance
        
        # 特征多样性（重要性分布的基尼系数）
        sorted_importance = np.sort(feature_importance_df['importance'].values)
        n = len(sorted_importance)
        gini_coefficient = (2 * np.arange(1, n + 1) - n - 1) @ sorted_importance / (n * sorted_importance.sum())
        
        return {
            'total_importance': total_importance,
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'top_5_importance': top_5_importance,
            'top_10_importance': top_10_importance,
            'concentration_ratio': concentration_ratio,
            'gini_coefficient': gini_coefficient,
            'feature_count': len(feature_importance_df)
        }
    
    def generate_metrics_report(self, metrics_dict, report_type='summary'):
        """
        生成指标报告
        
        Parameters:
        -----------
        metrics_dict : dict
            指标字典
        report_type : str
            报告类型：'summary', 'detailed'
            
        Returns:
        --------
        str : 格式化的报告
        """
        report = []
        
        if report_type == 'summary':
            report.append("模型性能指标摘要")
            report.append("=" * 40)
            
            # 主要指标
            main_metrics = ['r2', 'rmse', 'mae', 'accuracy', 'f1_score']
            for metric in main_metrics:
                if metric in metrics_dict:
                    value = metrics_dict[metric]
                    if isinstance(value, float):
                        report.append(f"{metric.upper()}: {value:.4f}")
                    else:
                        report.append(f"{metric.upper()}: {value}")
        
        elif report_type == 'detailed':
            report.append("详细模型性能指标")
            report.append("=" * 40)
            
            for metric_name, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    report.append(f"{metric_name}: {value:.4f}")
                elif isinstance(value, np.ndarray):
                    report.append(f"{metric_name}: {value.tolist()}")
                else:
                    report.append(f"{metric_name}: {value}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # 示例用法
    print("评估指标模块")
    print("请导入并使用 MetricsCalculator 类进行指标计算") 
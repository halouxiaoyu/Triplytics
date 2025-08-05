"""
模型评估模块

该模块提供对机器学习模型的评估功能，包括：
- 回归模型评估
- 分类模型评估
- 时间序列模型评估
- 模型比较和选择
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估类"""
    
    def __init__(self):
        """初始化模型评估器"""
        self.evaluation_results = {}
        
    def evaluate_regression_model(self, y_true, y_pred, model_name="model"):
        """
        评估回归模型
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        model_name : str
            模型名称
            
        Returns:
        --------
        dict : 评估结果
        """
        # 计算评估指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算相对误差
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 计算调整R²
        n = len(y_true)
        p = 1  # 假设特征数量为1，实际使用时需要传入
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        results = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'mape': mape
        }
        
        self.evaluation_results[model_name] = results
        
        # 打印结果
        print(f"\n{model_name} 回归模型评估结果:")
        print("-" * 50)
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")
        print(f"调整决定系数 (Adjusted R²): {adjusted_r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
        
        return results
    
    def evaluate_classification_model(self, y_true, y_pred, y_prob=None, model_name="model"):
        """
        评估分类模型
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        y_prob : array-like, optional
            预测概率
        model_name : str
            模型名称
            
        Returns:
        --------
        dict : 评估结果
        """
        # 基础分类指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # 如果有概率预测，计算ROC AUC
        if y_prob is not None:
            try:
                # 处理多分类情况
                if len(np.unique(y_true)) > 2:
                    # 多分类ROC AUC
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    if y_prob.shape[1] == len(np.unique(y_true)):
                        roc_auc = roc_auc_score(y_true_bin, y_prob, average='weighted')
                    else:
                        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                else:
                    # 二分类ROC AUC
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
                
                results['roc_auc'] = roc_auc
            except Exception as e:
                print(f"计算ROC AUC时出错: {e}")
                results['roc_auc'] = None
        
        self.evaluation_results[model_name] = results
        
        # 打印结果
        print(f"\n{model_name} 分类模型评估结果:")
        print("-" * 50)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-Score): {f1:.4f}")
        if 'roc_auc' in results and results['roc_auc'] is not None:
            print(f"ROC AUC: {results['roc_auc']:.4f}")
        
        return results
    
    def cross_validation_evaluation(self, model, X, y, cv=5, scoring='r2', model_name="model"):
        """
        交叉验证评估
        
        Parameters:
        -----------
        model : estimator
            机器学习模型
        X : array-like
            特征数据
        y : array-like
            目标数据
        cv : int
            交叉验证折数
        scoring : str
            评估指标
        model_name : str
            模型名称
            
        Returns:
        --------
        dict : 交叉验证结果
        """
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'model_name': model_name,
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        }
        
        self.evaluation_results[f"{model_name}_cv"] = results
        
        # 打印结果
        print(f"\n{model_name} 交叉验证结果 ({scoring}):")
        print("-" * 50)
        print(f"平均分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"分数范围: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        print(f"各折分数: {cv_scores}")
        
        return results
    
    def plot_regression_evaluation(self, y_true, y_pred, model_name="model"):
        """
        绘制回归模型评估图表
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        model_name : str
            模型名称
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测值 vs 真实值散点图
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title(f'{model_name} - 预测值 vs 真实值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title(f'{model_name} - 残差图')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差直方图
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title(f'{model_name} - 残差分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{model_name} - Q-Q图')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_classification_evaluation(self, y_true, y_pred, y_prob=None, model_name="model"):
        """
        绘制分类模型评估图表
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        y_prob : array-like, optional
            预测概率
        model_name : str
            模型名称
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('预测标签')
        axes[0, 0].set_ylabel('真实标签')
        axes[0, 0].set_title(f'{model_name} - 混淆矩阵')
        
        # 2. 分类报告
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # 创建分类报告热力图
        report_matrix = report_df.iloc[:-3, :-1].astype(float)  # 排除最后三行和最后一列
        sns.heatmap(report_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 1])
        axes[0, 1].set_title(f'{model_name} - 分类报告')
        
        # 3. ROC曲线（如果有概率预测）
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # 二分类ROC曲线
                    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
                    
                    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                                   label=f'ROC curve (AUC = {roc_auc:.2f})')
                    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    axes[1, 0].set_xlim([0.0, 1.0])
                    axes[1, 0].set_ylim([0.0, 1.05])
                    axes[1, 0].set_xlabel('假正率')
                    axes[1, 0].set_ylabel('真正率')
                    axes[1, 0].set_title(f'{model_name} - ROC曲线')
                    axes[1, 0].legend(loc="lower right")
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    # 多分类ROC曲线
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    n_classes = len(np.unique(y_true))
                    
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                        roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                        axes[1, 0].plot(fpr, tpr, lw=2, 
                                       label=f'Class {i} (AUC = {roc_auc:.2f})')
                    
                    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    axes[1, 0].set_xlim([0.0, 1.0])
                    axes[1, 0].set_ylim([0.0, 1.05])
                    axes[1, 0].set_xlabel('假正率')
                    axes[1, 0].set_ylabel('真正率')
                    axes[1, 0].set_title(f'{model_name} - ROC曲线')
                    axes[1, 0].legend(loc="lower right")
                    axes[1, 0].grid(True, alpha=0.3)
            except Exception as e:
                print(f"绘制ROC曲线时出错: {e}")
                axes[1, 0].text(0.5, 0.5, 'ROC曲线不可用', ha='center', va='center')
                axes[1, 0].set_title(f'{model_name} - ROC曲线')
        
        # 4. 预测概率分布
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # 二分类概率分布
                    prob_positive = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob
                    axes[1, 1].hist(prob_positive[y_true == 0], bins=30, alpha=0.5, 
                                   label='负类', color='red')
                    axes[1, 1].hist(prob_positive[y_true == 1], bins=30, alpha=0.5, 
                                   label='正类', color='blue')
                    axes[1, 1].set_xlabel('预测概率')
                    axes[1, 1].set_ylabel('频数')
                    axes[1, 1].set_title(f'{model_name} - 预测概率分布')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    # 多分类概率分布
                    for i in range(len(np.unique(y_true))):
                        axes[1, 1].hist(y_prob[:, i], bins=30, alpha=0.5, 
                                       label=f'Class {i}')
                    axes[1, 1].set_xlabel('预测概率')
                    axes[1, 1].set_ylabel('频数')
                    axes[1, 1].set_title(f'{model_name} - 预测概率分布')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                print(f"绘制概率分布时出错: {e}")
                axes[1, 1].text(0.5, 0.5, '概率分布不可用', ha='center', va='center')
                axes[1, 1].set_title(f'{model_name} - 预测概率分布')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def compare_models(self, metric='r2'):
        """
        比较多个模型
        
        Parameters:
        -----------
        metric : str
            比较指标
            
        Returns:
        --------
        pandas.DataFrame : 模型比较结果
        """
        if not self.evaluation_results:
            raise ValueError("没有评估结果可供比较")
            
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            if metric in results:
                comparison_data.append({
                    'Model': model_name,
                    metric.upper(): results[metric]
                })
        
        if not comparison_data:
            raise ValueError(f"没有找到指标 {metric} 的评估结果")
            
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(metric.upper(), ascending=False)
        
        # 打印比较结果
        print(f"\n模型比较结果 ({metric.upper()}):")
        print("-" * 50)
        print(comparison_df.to_string(index=False))
        
        # 绘制比较图表
        plt.figure(figsize=(10, 6))
        plt.bar(comparison_df['Model'], comparison_df[metric.upper()])
        plt.xlabel('模型')
        plt.ylabel(metric.upper())
        plt.title(f'模型性能比较 - {metric.upper()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def generate_evaluation_report(self, output_path=None):
        """
        生成评估报告
        
        Parameters:
        -----------
        output_path : str, optional
            报告保存路径
        """
        if not self.evaluation_results:
            raise ValueError("没有评估结果可供报告")
            
        print("=" * 60)
        print("模型评估报告")
        print("=" * 60)
        
        # 统计模型数量
        n_models = len(self.evaluation_results)
        print(f"\n1. 评估概览:")
        print(f"   评估模型数量: {n_models}")
        
        # 分类模型和回归模型
        regression_models = []
        classification_models = []
        
        for model_name, results in self.evaluation_results.items():
            if 'r2' in results:
                regression_models.append(model_name)
            elif 'accuracy' in results:
                classification_models.append(model_name)
        
        print(f"   回归模型: {len(regression_models)}")
        print(f"   分类模型: {len(classification_models)}")
        
        # 最佳模型
        print(f"\n2. 最佳模型:")
        
        if regression_models:
            best_regression = max(regression_models, 
                                key=lambda x: self.evaluation_results[x].get('r2', 0))
            best_r2 = self.evaluation_results[best_regression].get('r2', 0)
            print(f"   最佳回归模型: {best_regression} (R² = {best_r2:.4f})")
        
        if classification_models:
            best_classification = max(classification_models, 
                                    key=lambda x: self.evaluation_results[x].get('accuracy', 0))
            best_accuracy = self.evaluation_results[best_classification].get('accuracy', 0)
            print(f"   最佳分类模型: {best_classification} (Accuracy = {best_accuracy:.4f})")
        
        # 详细结果
        print(f"\n3. 详细评估结果:")
        print("-" * 50)
        
        for model_name, results in self.evaluation_results.items():
            print(f"\n{model_name}:")
            for metric, value in results.items():
                if metric != 'model_name':
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
        print("\n" + "=" * 60)
        print("报告生成完成")
        print("=" * 60)


if __name__ == "__main__":
    # 示例用法
    print("模型评估模块")
    print("请导入并使用 ModelEvaluator 类进行模型评估") 
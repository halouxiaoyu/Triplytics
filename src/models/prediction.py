"""
预测模型模块

该模块提供对滴滴出行数据的预测分析功能，包括：
- 需求预测
- 时间序列预测
- 回归模型
- 模型评估和比较
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class DemandPredictor:
    """需求预测类"""
    
    def __init__(self, df=None):
        """
        初始化需求预测器
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            包含预测数据的DataFrame
        """
        self.df = df
        self.feature_columns = None
        self.target_column = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        
    def set_data(self, df, feature_columns, target_column):
        """
        设置预测数据
        
        Parameters:
        -----------
        df : pandas.DataFrame
            数据DataFrame
        feature_columns : list
            特征列名列表
        target_column : str
            目标列名
        """
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # 数据清洗
        self.df = self.df.dropna(subset=[target_column])
        
        # 确保所有特征列存在
        missing_cols = [col for col in feature_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"缺失特征列: {missing_cols}")
            
        print(f"数据设置完成，特征数量: {len(feature_columns)}, 样本数量: {len(self.df)}")
        
    def prepare_features(self, test_size=0.2, random_state=42):
        """
        准备训练和测试数据
        
        Parameters:
        -----------
        test_size : float
            测试集比例
        random_state : int
            随机种子
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            raise ValueError("请先设置数据")
            
        # 准备特征和目标
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        
        # 处理分类特征
        X = self._encode_categorical_features(X)
        
        # 处理缺失值
        X = self._handle_missing_values(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _encode_categorical_features(self, X):
        """编码分类特征"""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                X_encoded[col] = pd.Categorical(X_encoded[col]).codes
                
        return X_encoded
    
    def _handle_missing_values(self, X):
        """处理缺失值"""
        X_filled = X.copy()
        
        # 数值列用中位数填充
        numeric_cols = X_filled.select_dtypes(include=[np.number]).columns
        X_filled[numeric_cols] = X_filled[numeric_cols].fillna(X_filled[numeric_cols].median())
        
        # 分类列用众数填充
        categorical_cols = X_filled.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0])
            
        return X_filled
    
    def train_linear_regression(self, **kwargs):
        """
        训练线性回归模型
        
        Parameters:
        -----------
        **kwargs : dict
            线性回归参数
            
        Returns:
        --------
        dict : 模型结果
        """
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # 训练模型
        model = LinearRegression(**kwargs)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        result = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'metrics': metrics
        }
        
        self.models['linear_regression'] = result
        
        print(f"线性回归训练完成:")
        print(f"  训练集 R²: {metrics['train_r2']:.4f}")
        print(f"  测试集 R²: {metrics['test_r2']:.4f}")
        print(f"  测试集 RMSE: {metrics['test_rmse']:.4f}")
        
        return result
    
    def train_random_forest(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        """
        训练随机森林模型
        
        Parameters:
        -----------
        n_estimators : int
            树的数量
        max_depth : int
            最大深度
        random_state : int
            随机种子
        **kwargs : dict
            其他随机森林参数
            
        Returns:
        --------
        dict : 模型结果
        """
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # 训练模型
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        self.models['random_forest'] = result
        self.feature_importance['random_forest'] = feature_importance
        
        print(f"随机森林训练完成:")
        print(f"  训练集 R²: {metrics['train_r2']:.4f}")
        print(f"  测试集 R²: {metrics['test_r2']:.4f}")
        print(f"  测试集 RMSE: {metrics['test_rmse']:.4f}")
        
        return result
    
    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, **kwargs):
        """
        训练XGBoost模型
        
        Parameters:
        -----------
        n_estimators : int
            树的数量
        max_depth : int
            最大深度
        learning_rate : float
            学习率
        random_state : int
            随机种子
        **kwargs : dict
            其他XGBoost参数
            
        Returns:
        --------
        dict : 模型结果
        """
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # 训练模型
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        self.models['xgboost'] = result
        self.feature_importance['xgboost'] = feature_importance
        
        print(f"XGBoost训练完成:")
        print(f"  训练集 R²: {metrics['train_r2']:.4f}")
        print(f"  测试集 R²: {metrics['test_r2']:.4f}")
        print(f"  测试集 RMSE: {metrics['test_rmse']:.4f}")
        
        return result
    
    def train_lightgbm(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, **kwargs):
        """
        训练LightGBM模型
        
        Parameters:
        -----------
        n_estimators : int
            树的数量
        max_depth : int
            最大深度
        learning_rate : float
            学习率
        random_state : int
            随机种子
        **kwargs : dict
            其他LightGBM参数
            
        Returns:
        --------
        dict : 模型结果
        """
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # 训练模型
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        self.models['lightgbm'] = result
        self.feature_importance['lightgbm'] = feature_importance
        
        print(f"LightGBM训练完成:")
        print(f"  训练集 R²: {metrics['train_r2']:.4f}")
        print(f"  测试集 R²: {metrics['test_r2']:.4f}")
        print(f"  测试集 RMSE: {metrics['test_rmse']:.4f}")
        
        return result
    
    def train_prophet(self, time_column, forecast_periods=24, **kwargs):
        """
        训练Prophet模型
        
        Parameters:
        -----------
        time_column : str
            时间列名
        forecast_periods : int
            预测期数
        **kwargs : dict
            其他Prophet参数
            
        Returns:
        --------
        dict : 模型结果
        """
        if time_column not in self.df.columns:
            raise ValueError(f"时间列 {time_column} 不存在")
            
        # 准备Prophet数据格式
        df_prophet = self.df[[time_column, self.target_column]].copy()
        df_prophet.columns = ['ds', 'y']
        
        # 确保时间格式正确
        if not pd.api.types.is_datetime64_any_dtype(df_prophet['ds']):
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            
        # 按时间排序
        df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
        
        # 训练模型
        model = Prophet(**kwargs)
        model.fit(df_prophet)
        
        # 预测
        future = model.make_future_dataframe(periods=forecast_periods, freq='H')
        forecast = model.predict(future)
        
        # 评估（使用历史数据）
        historical_forecast = forecast[forecast['ds'].isin(df_prophet['ds'])]
        y_true = df_prophet['y'].values
        y_pred = historical_forecast['yhat'].values
        
        metrics = {
            'train_r2': r2_score(y_true, y_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'train_mae': mean_absolute_error(y_true, y_pred),
            'test_r2': None,  # Prophet不区分训练测试集
            'test_rmse': None,
            'test_mae': None
        }
        
        result = {
            'model': model,
            'forecast': forecast,
            'metrics': metrics
        }
        
        self.models['prophet'] = result
        
        print(f"Prophet训练完成:")
        print(f"  R²: {metrics['train_r2']:.4f}")
        print(f"  RMSE: {metrics['train_rmse']:.4f}")
        print(f"  MAE: {metrics['train_mae']:.4f}")
        
        return result
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """计算评估指标"""
        return {
            'train_r2': r2_score(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
    
    def compare_models(self):
        """
        比较所有训练的模型
        
        Returns:
        --------
        pandas.DataFrame : 模型比较结果
        """
        if not self.models:
            raise ValueError("请先训练至少一个模型")
            
        comparison_data = []
        
        for model_name, result in self.models.items():
            metrics = result['metrics']
            
            comparison_data.append({
                'Model': model_name,
                'Train_R2': metrics.get('train_r2', 'N/A'),
                'Test_R2': metrics.get('test_r2', 'N/A'),
                'Train_RMSE': metrics.get('train_rmse', 'N/A'),
                'Test_RMSE': metrics.get('test_rmse', 'N/A'),
                'Train_MAE': metrics.get('train_mae', 'N/A'),
                'Test_MAE': metrics.get('test_mae', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 打印比较结果
        print("\n模型性能比较:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_feature_importance(self, model_name='random_forest', top_n=10):
        """
        绘制特征重要性
        
        Parameters:
        -----------
        model_name : str
            模型名称
        top_n : int
            显示前N个重要特征
        """
        if model_name not in self.feature_importance:
            raise ValueError(f"模型 {model_name} 没有特征重要性信息")
            
        feature_importance = self.feature_importance[model_name]
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('重要性')
        plt.title(f'{model_name.upper()} 特征重要性 (Top {top_n})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def plot_predictions(self, model_name='random_forest'):
        """
        绘制预测结果
        
        Parameters:
        -----------
        model_name : str
            模型名称
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
            
        result = self.models[model_name]
        
        if 'predictions' not in result:
            print(f"模型 {model_name} 不支持预测结果可视化")
            return
            
        predictions = result['predictions']
        y_train_pred = predictions['train']
        y_test_pred = predictions['test']
        
        # 获取原始数据
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练集预测
        axes[0].scatter(y_train, y_train_pred, alpha=0.5)
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        axes[0].set_xlabel('实际值')
        axes[0].set_ylabel('预测值')
        axes[0].set_title('训练集预测结果')
        axes[0].grid(True, alpha=0.3)
        
        # 测试集预测
        axes[1].scatter(y_test, y_test_pred, alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('实际值')
        axes[1].set_ylabel('预测值')
        axes[1].set_title('测试集预测结果')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def predict_demand(self, X_new, model_name='random_forest'):
        """
        使用训练好的模型进行预测
        
        Parameters:
        -----------
        X_new : pandas.DataFrame
            新数据
        model_name : str
            模型名称
            
        Returns:
        --------
        numpy.ndarray : 预测结果
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
            
        model = self.models[model_name]['model']
        
        # 数据预处理
        X_new_processed = self._encode_categorical_features(X_new)
        X_new_processed = self._handle_missing_values(X_new_processed)
        X_new_scaled = self.scaler.transform(X_new_processed)
        
        # 预测
        predictions = model.predict(X_new_scaled)
        
        return predictions
    
    def generate_prediction_report(self, output_path=None):
        """
        生成预测分析报告
        
        Parameters:
        -----------
        output_path : str, optional
            报告保存路径
        """
        if not self.models:
            raise ValueError("请先训练至少一个模型")
            
        print("=" * 60)
        print("需求预测分析报告")
        print("=" * 60)
        
        # 数据概览
        print(f"\n1. 数据概览:")
        print(f"   特征数量: {len(self.feature_columns)}")
        print(f"   样本数量: {len(self.df)}")
        print(f"   目标变量: {self.target_column}")
        
        # 模型性能比较
        print(f"\n2. 模型性能比较:")
        comparison_df = self.compare_models()
        
        # 最佳模型
        best_model = comparison_df.loc[comparison_df['Test_R2'].idxmax(), 'Model']
        print(f"\n3. 最佳模型:")
        print(f"   基于测试集R²，最佳模型: {best_model}")
        
        # 特征重要性（如果有）
        if self.feature_importance:
            print(f"\n4. 特征重要性分析:")
            for model_name in self.feature_importance.keys():
                if model_name in self.models:
                    top_features = self.feature_importance[model_name].head(5)
                    print(f"   {model_name.upper()} 前5个重要特征:")
                    for _, row in top_features.iterrows():
                        print(f"     {row['feature']}: {row['importance']:.4f}")
        
        print("\n" + "=" * 60)
        print("报告生成完成")
        print("=" * 60)


if __name__ == "__main__":
    # 示例用法
    print("需求预测模块")
    print("请导入并使用 DemandPredictor 类进行需求预测") 
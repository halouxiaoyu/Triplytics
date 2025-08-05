"""
高级机器学习模型模块
包含深度学习、集成学习、强化学习等高级模型
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 深度学习库
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# 强化学习库
try:
    import gym
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

from config.logging_config import LoggerMixin, log_execution_time
from src.models.evaluation import ModelEvaluator

class AdvancedMLModels(LoggerMixin):
    """高级机器学习模型类"""
    
    def __init__(self):
        """初始化高级模型"""
        super().__init__()
        self.models = {}
        self.scalers = {}
        self.evaluator = ModelEvaluator()
        
    @log_execution_time
    def create_ensemble_model(self, base_models=None, voting_method='soft'):
        """
        创建集成学习模型
        
        Args:
            base_models: 基础模型列表
            voting_method: 投票方法 ('soft', 'hard')
            
        Returns:
            集成模型
        """
        if base_models is None:
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
                ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42)),
                ('cat', CatBoostRegressor(iterations=100, random_state=42, verbose=False))
            ]
        
        ensemble = VotingRegressor(
            estimators=base_models,
            voting=voting_method
        )
        
        self.models['ensemble'] = ensemble
        self.log_info(f"集成模型创建完成，包含 {len(base_models)} 个基础模型")
        return ensemble
    
    @log_execution_time
    def create_deep_learning_model(self, input_dim, model_type='mlp'):
        """
        创建深度学习模型
        
        Args:
            input_dim: 输入特征维度
            model_type: 模型类型 ('mlp', 'lstm', 'gru')
            
        Returns:
            深度学习模型
        """
        if not TENSORFLOW_AVAILABLE:
            self.log_warning("TensorFlow不可用，使用MLPRegressor替代")
            return MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        
        if model_type == 'mlp':
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
        elif model_type == 'lstm':
            # 需要3D输入 (samples, timesteps, features)
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(None, input_dim)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
        elif model_type == 'gru':
            model = Sequential([
                GRU(64, return_sequences=True, input_shape=(None, input_dim)),
                Dropout(0.2),
                GRU(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        self.models[f'deep_{model_type}'] = model
        self.log_info(f"深度学习模型创建完成，类型: {model_type}")
        return model
    
    @log_execution_time
    def create_autoencoder(self, input_dim, encoding_dim=32):
        """
        创建自编码器用于特征降维和异常检测
        
        Args:
            input_dim: 输入特征维度
            encoding_dim: 编码维度
            
        Returns:
            自编码器模型
        """
        if not TENSORFLOW_AVAILABLE:
            self.log_warning("TensorFlow不可用，无法创建自编码器")
            return None
        
        # 编码器
        input_layer = tf.keras.Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # 解码器
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # 自编码器
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # 编码器
        encoder = Model(input_layer, encoded)
        
        self.models['autoencoder'] = autoencoder
        self.models['encoder'] = encoder
        self.log_info(f"自编码器创建完成，编码维度: {encoding_dim}")
        return autoencoder, encoder
    
    @log_execution_time
    def create_reinforcement_learning_model(self, env_name='Taxi-v3'):
        """
        创建强化学习模型
        
        Args:
            env_name: 环境名称
            
        Returns:
            强化学习模型
        """
        if not RL_AVAILABLE:
            self.log_warning("强化学习库不可用")
            return None
        
        try:
            # 创建环境
            env = gym.make(env_name)
            env = DummyVecEnv([lambda: env])
            
            # 创建PPO模型
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
            
            self.models['rl_ppo'] = model
            self.log_info(f"强化学习模型创建完成，环境: {env_name}")
            return model
            
        except Exception as e:
            self.log_error(f"强化学习模型创建失败: {str(e)}")
            return None
    
    @log_execution_time
    def train_models(self, X_train, y_train, X_val=None, y_val=None, 
                    models_to_train=None):
        """
        训练多个模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            models_to_train: 要训练的模型列表
            
        Returns:
            训练结果字典
        """
        if models_to_train is None:
            models_to_train = ['ensemble', 'deep_mlp']
        
        results = {}
        
        for model_name in models_to_train:
            if model_name not in self.models:
                self.log_warning(f"模型 {model_name} 不存在，跳过")
                continue
            
            self.log_info(f"开始训练模型: {model_name}")
            
            try:
                model = self.models[model_name]
                
                if model_name.startswith('deep_'):
                    # 深度学习模型
                    callbacks = [
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=5)
                    ]
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val) if X_val is not None else None,
                        epochs=100,
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    results[model_name] = {
                        'model': model,
                        'history': history.history if hasattr(history, 'history') else None
                    }
                    
                elif model_name.startswith('rl_'):
                    # 强化学习模型
                    model.learn(total_timesteps=10000)
                    results[model_name] = {'model': model}
                    
                else:
                    # 传统机器学习模型
                    model.fit(X_train, y_train)
                    results[model_name] = {'model': model}
                
                self.log_info(f"模型 {model_name} 训练完成")
                
            except Exception as e:
                self.log_error(f"模型 {model_name} 训练失败: {str(e)}")
                continue
        
        return results
    
    @log_execution_time
    def hyperparameter_optimization(self, X_train, y_train, model_type='ensemble'):
        """
        超参数优化
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            model_type: 模型类型
            
        Returns:
            最佳模型
        """
        if model_type == 'ensemble':
            # 随机森林超参数优化
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.models['optimized_rf'] = grid_search.best_estimator_
            self.log_info(f"最佳参数: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        elif model_type == 'xgboost':
            # XGBoost超参数优化
            xgb_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                xgb_model, xgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.models['optimized_xgb'] = grid_search.best_estimator_
            self.log_info(f"最佳参数: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return None
    
    @log_execution_time
    def feature_importance_analysis(self, model_name='ensemble'):
        """
        特征重要性分析
        
        Args:
            model_name: 模型名称
            
        Returns:
            特征重要性字典
        """
        if model_name not in self.models:
            self.log_error(f"模型 {model_name} 不存在")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # 随机森林、XGBoost等模型
            return dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
        
        elif hasattr(model, 'coef_'):
            # 线性模型
            return dict(zip(range(len(model.coef_)), np.abs(model.coef_)))
        
        elif model_name.startswith('deep_'):
            # 深度学习模型 - 使用梯度分析
            self.log_info("深度学习模型特征重要性分析需要额外实现")
            return None
        
        return None
    
    @log_execution_time
    def anomaly_detection(self, X, method='autoencoder', threshold=0.95):
        """
        异常检测
        
        Args:
            X: 输入数据
            method: 检测方法 ('autoencoder', 'isolation_forest', 'one_class_svm')
            threshold: 异常阈值
            
        Returns:
            异常检测结果
        """
        if method == 'autoencoder' and 'autoencoder' in self.models:
            # 使用自编码器进行异常检测
            reconstructed = self.models['autoencoder'].predict(X)
            mse = np.mean(np.power(X - reconstructed, 2), axis=1)
            threshold_value = np.percentile(mse, threshold * 100)
            anomalies = mse > threshold_value
            
            return {
                'anomalies': anomalies,
                'scores': mse,
                'threshold': threshold_value
            }
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=1-threshold, random_state=42)
            predictions = iso_forest.fit_predict(X)
            anomalies = predictions == -1
            
            return {
                'anomalies': anomalies,
                'scores': iso_forest.decision_function(X),
                'threshold': 0
            }
        
        return None
    
    @log_execution_time
    def model_interpretation(self, model_name='ensemble', X_sample=None):
        """
        模型解释
        
        Args:
            model_name: 模型名称
            X_sample: 样本数据
            
        Returns:
            解释结果
        """
        if model_name not in self.models:
            self.log_error(f"模型 {model_name} 不存在")
            return None
        
        model = self.models[model_name]
        
        # 特征重要性
        importance = self.feature_importance_analysis(model_name)
        
        # 模型复杂度
        complexity = {
            'parameters': self._count_parameters(model),
            'type': type(model).__name__
        }
        
        # 预测解释（如果使用SHAP）
        try:
            import shap
            explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else None
            if explainer and X_sample is not None:
                shap_values = explainer.shap_values(X_sample)
                return {
                    'importance': importance,
                    'complexity': complexity,
                    'shap_values': shap_values
                }
        except ImportError:
            self.log_warning("SHAP不可用，跳过SHAP解释")
        
        return {
            'importance': importance,
            'complexity': complexity
        }
    
    def _count_parameters(self, model):
        """计算模型参数数量"""
        if hasattr(model, 'n_parameters'):
            return model.n_parameters
        elif hasattr(model, 'coef_'):
            return len(model.coef_)
        elif hasattr(model, 'feature_importances_'):
            return len(model.feature_importances_)
        elif TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
            return model.count_params()
        else:
            return "未知"
    
    @log_execution_time
    def ensemble_prediction(self, X, weights=None):
        """
        集成预测
        
        Args:
            X: 输入特征
            weights: 模型权重
            
        Returns:
            集成预测结果
        """
        predictions = {}
        
        for name, model in self.models.items():
            if name.startswith('deep_'):
                # 深度学习模型
                pred = model.predict(X, verbose=0).flatten()
            elif name.startswith('rl_'):
                # 强化学习模型
                pred = model.predict(X)
            else:
                # 传统机器学习模型
                pred = model.predict(X)
            
            predictions[name] = pred
        
        if weights is None:
            # 等权重平均
            weights = {name: 1.0/len(predictions) for name in predictions.keys()}
        
        # 加权平均
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += weights.get(name, 0) * pred
        
        return ensemble_pred, predictions
    
    def save_models(self, filepath):
        """保存模型"""
        import joblib
        
        # 保存传统机器学习模型
        ml_models = {k: v for k, v in self.models.items() 
                    if not k.startswith('deep_') and not k.startswith('rl_')}
        joblib.dump(ml_models, f"{filepath}_ml_models.pkl")
        
        # 保存深度学习模型
        if TENSORFLOW_AVAILABLE:
            deep_models = {k: v for k, v in self.models.items() if k.startswith('deep_')}
            for name, model in deep_models.items():
                model.save(f"{filepath}_{name}.h5")
        
        # 保存强化学习模型
        if RL_AVAILABLE:
            rl_models = {k: v for k, v in self.models.items() if k.startswith('rl_')}
            for name, model in rl_models.items():
                model.save(f"{filepath}_{name}")
        
        self.log_info(f"模型已保存到: {filepath}")
    
    def load_models(self, filepath):
        """加载模型"""
        import joblib
        
        # 加载传统机器学习模型
        try:
            ml_models = joblib.load(f"{filepath}_ml_models.pkl")
            self.models.update(ml_models)
        except FileNotFoundError:
            self.log_warning("未找到传统机器学习模型文件")
        
        # 加载深度学习模型
        if TENSORFLOW_AVAILABLE:
            import glob
            deep_model_files = glob.glob(f"{filepath}_deep_*.h5")
            for file in deep_model_files:
                name = file.split('_')[-1].replace('.h5', '')
                self.models[f'deep_{name}'] = tf.keras.models.load_model(file)
        
        # 加载强化学习模型
        if RL_AVAILABLE:
            import glob
            rl_model_files = glob.glob(f"{filepath}_rl_*")
            for file in rl_model_files:
                name = file.split('_')[-1]
                self.models[f'rl_{name}'] = PPO.load(file)
        
        self.log_info(f"模型已从 {filepath} 加载")

def create_advanced_model_pipeline():
    """创建高级模型流水线"""
    return AdvancedMLModels()

if __name__ == "__main__":
    # 示例用法
    models = AdvancedMLModels()
    print("高级机器学习模型模块已加载") 
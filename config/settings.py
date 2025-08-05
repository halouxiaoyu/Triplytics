"""
项目配置文件
包含所有重要的配置参数和设置
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR  # 修改为直接在data目录下查找
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# 报告目录
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PRESENTATIONS_DIR = REPORTS_DIR / "presentations"

# 模型保存目录
MODELS_DIR = PROJECT_ROOT / "models"

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_LEVEL = "INFO"

# 数据文件配置
DATA_FILES = {
    "uber_april": "uber-raw-data-apr14.csv",
    "uber_august": "uber-raw-data-aug14.csv",
    "uber_july": "uber-raw-data-jul14.csv",
    "uber_june": "uber-raw-data-jun14.csv",
    "uber_may": "uber-raw-data-may14.csv",
    "uber_september": "uber-raw-data-sep14.csv",
    "uber_jan_june": "uber-raw-data-janjune-15.csv",
    "uber_jan_feb": "Uber-Jan-Feb-FOIL.csv"
}

# 数据列配置
COLUMN_MAPPING = {
    "Date/Time": "datetime",
    "Lat": "latitude",
    "Lon": "longitude",
    "Base": "base_station"
}

# 时间配置
TIMEZONE = "America/New_York"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

# 地理配置
NYC_BOUNDS = {
    "min_lat": 40.5,
    "max_lat": 41.0,
    "min_lon": -74.3,
    "max_lon": -73.7
}

# 聚类配置
CLUSTERING_CONFIG = {
    "kmeans": {
        "n_clusters": 8,
        "random_state": 42,
        "n_init": 10
    },
    "dbscan": {
        "eps": 0.01,
        "min_samples": 50
    }
}

# 机器学习模型配置
ML_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    }
}

# 时间序列配置
TIME_SERIES_CONFIG = {
    "arima": {
        "order": (2, 1, 2),
        "seasonal_order": (1, 1, 1, 24)
    },
    "prophet": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0
    }
}

# 特征工程配置
FEATURE_CONFIG = {
    "time_features": [
        "hour", "day", "weekday", "month", "quarter",
        "is_weekend", "is_peak_hour", "is_holiday"
    ],
    "spatial_features": [
        "cluster_id", "distance_to_center", "neighborhood"
    ],
    "lag_features": [1, 2, 3, 6, 12, 24],
    "rolling_features": [3, 6, 12, 24]
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "style": "seaborn-v0_8",
    "figure_size": (12, 8),
    "dpi": 300,
    "color_palette": "viridis",
    "font_size": 12
}

# 评估指标配置
EVALUATION_METRICS = [
    "mae", "mse", "rmse", "mape", "r2_score",
    "precision", "recall", "f1_score", "auc"
]

# 交叉验证配置
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42
}

# 数据采样配置
SAMPLING_CONFIG = {
    "sample_size": 50000,
    "random_state": 42,
    "stratify": None
}

# 异常检测配置
ANOMALY_DETECTION_CONFIG = {
    "isolation_forest": {
        "contamination": 0.1,
        "random_state": 42
    },
    "one_class_svm": {
        "nu": 0.1,
        "kernel": "rbf"
    }
}

# 数据库配置
DATABASE_CONFIG = {
    "sqlite": {
        "path": str(PROCESSED_DATA_DIR / "didi.db")
    },
    "postgresql": {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "didi"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "")
    }
}

# API配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "reload": True
}

# 缓存配置
CACHE_CONFIG = {
    "enabled": True,
    "ttl": 3600,  # 1小时
    "max_size": 1000
}

# 并行处理配置
PARALLEL_CONFIG = {
    "n_jobs": -1,
    "backend": "multiprocessing",
    "batch_size": 1000
}

# 模型保存配置
MODEL_SAVE_CONFIG = {
    "format": "pickle",
    "compress": True,
    "version": "1.0.0"
}

# 日志轮转配置
LOG_ROTATION_CONFIG = {
    "max_size": "10MB",
    "retention": "30 days",
    "compression": "zip"
}

# 环境变量配置
ENV_VARS = {
    "DATA_PATH": str(DATA_DIR),
    "MODELS_PATH": str(MODELS_DIR),
    "REPORTS_PATH": str(REPORTS_DIR),
    "LOG_LEVEL": LOG_LEVEL
}

def get_config() -> Dict[str, Any]:
    """
    获取完整配置字典
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dirs": {
            "raw": str(RAW_DATA_DIR),
            "processed": str(PROCESSED_DATA_DIR),
            "external": str(EXTERNAL_DATA_DIR)
        },
        "reports_dir": str(REPORTS_DIR),
        "models_dir": str(MODELS_DIR),
        "clustering": CLUSTERING_CONFIG,
        "ml_models": ML_CONFIG,
        "time_series": TIME_SERIES_CONFIG,
        "features": FEATURE_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "evaluation": EVALUATION_METRICS,
        "cv": CV_CONFIG,
        "sampling": SAMPLING_CONFIG,
        "anomaly_detection": ANOMALY_DETECTION_CONFIG,
        "database": DATABASE_CONFIG,
        "api": API_CONFIG,
        "cache": CACHE_CONFIG,
        "parallel": PARALLEL_CONFIG,
        "model_save": MODEL_SAVE_CONFIG,
        "log_rotation": LOG_ROTATION_CONFIG
    }

def create_directories():
    """
    创建必要的目录结构
    """
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        REPORTS_DIR, FIGURES_DIR, PRESENTATIONS_DIR, MODELS_DIR, LOG_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directories()

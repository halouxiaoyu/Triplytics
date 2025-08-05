"""
日志配置模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
import json
from datetime import datetime

from .settings import LOG_DIR, LOG_LEVEL, LOG_ROTATION_CONFIG

class InterceptHandler(logging.Handler):
    """
    拦截标准库日志并重定向到loguru
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging(
    log_file: Optional[str] = None,
    level: str = LOG_LEVEL,
    rotation: str = LOG_ROTATION_CONFIG["max_size"],
    retention: str = LOG_ROTATION_CONFIG["retention"],
    compression: str = LOG_ROTATION_CONFIG["compression"],
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                 "<level>{level: <8}</level> | "
                 "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                 "<level>{message}</level>"
) -> None:
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        rotation: 日志轮转大小
        retention: 日志保留时间
        compression: 日志压缩格式
        format: 日志格式
    """
    # 确保日志目录存在
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 移除默认的处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format=format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件处理器
    if log_file is None:
        log_file = LOG_DIR / f"didi_{datetime.now().strftime('%Y%m%d')}.log"
    
    logger.add(
        log_file,
        format=format,
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=True,
        diagnose=True,
        enqueue=True  # 异步写入
    )
    
    # 拦截标准库日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # 设置第三方库的日志级别
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

def get_logger(name: str = __name__):
    """
    获取logger实例
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
    """
    return logger.bind(name=name)

class LoggerMixin:
    """
    日志混入类，为其他类提供日志功能
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """记录信息日志"""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """记录错误日志"""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, **kwargs)
    
    def log_exception(self, message: str, **kwargs):
        """记录异常日志"""
        self.logger.exception(message, **kwargs)

def log_function_call(func):
    """
    函数调用日志装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.info(f"Calling function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.exception(f"Function {func.__name__} failed with error: {str(e)}")
            raise
    return wrapper

def log_execution_time(func):
    """
    执行时间日志装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        logger.info(f"Starting execution of {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"Function {func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.exception(f"Function {func.__name__} failed after {execution_time:.2f} seconds")
            raise
    return wrapper

class PerformanceLogger:
    """
    性能日志记录器
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            end_time = datetime.now()
            execution_time = (end_time - self.start_time).total_seconds()
            if exc_type is None:
                self.logger.info(f"{self.name} completed in {execution_time:.2f} seconds")
            else:
                self.logger.error(f"{self.name} failed after {execution_time:.2f} seconds")

def log_data_info(df, name: str = "DataFrame"):
    """
    记录数据框信息
    
    Args:
        df: pandas DataFrame
        name: 数据框名称
    """
    logger = get_logger("data_info")
    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Data types:\n{df.dtypes}")
    logger.info(f"  Missing values:\n{df.isnull().sum()}")

def log_model_performance(model_name: str, metrics: dict):
    """
    记录模型性能
    
    Args:
        model_name: 模型名称
        metrics: 性能指标字典
    """
    logger = get_logger("model_performance")
    logger.info(f"Model: {model_name}")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

# 初始化日志配置
setup_logging()

# 导出主要函数
__all__ = [
    'setup_logging',
    'get_logger',
    'LoggerMixin',
    'log_function_call',
    'log_execution_time',
    'PerformanceLogger',
    'log_data_info',
    'log_model_performance'
]

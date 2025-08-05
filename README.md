# 滴滴出行数据分析平台

## 项目概述

这是一个专业的滴滴出行数据分析平台，基于Uber纽约数据集进行深度分析和建模。项目采用模块化设计，包含完整的数据分析流程，从数据预处理到高级机器学习模型。

## 项目架构

```
didi/
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── config/                   # 配置文件目录
│   ├── __init__.py
│   ├── settings.py          # 项目配置
│   └── logging_config.py    # 日志配置
├── data/                     # 数据目录
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后数据
│   └── external/            # 外部数据
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── data/                # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py   # 数据加载器
│   │   ├── data_cleaner.py  # 数据清洗
│   │   └── feature_engineer.py # 特征工程
│   ├── analysis/            # 分析模块
│   │   ├── __init__.py
│   │   ├── eda.py          # 探索性数据分析
│   │   ├── spatial_analysis.py # 空间分析
│   │   └── temporal_analysis.py # 时间序列分析
│   ├── models/              # 模型模块
│   │   ├── __init__.py
│   │   ├── clustering.py   # 聚类模型
│   │   ├── prediction.py   # 预测模型
│   │   └── evaluation.py   # 模型评估
│   ├── visualization/       # 可视化模块
│   │   ├── __init__.py
│   │   ├── charts.py       # 图表生成
│   │   ├── maps.py         # 地图可视化
│   │   └── dashboard.py    # 仪表盘
│   └── utils/               # 工具模块
│       ├── __init__.py
│       ├── helpers.py      # 辅助函数
│       └── metrics.py      # 评估指标
├── reports/                 # 报告目录
│   ├── figures/            # 图表
│   └── presentations/      # 演示文稿
├── tests/                   # 测试目录
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_models.py
└── scripts/                 # 脚本目录
    ├── run_analysis.py     # 运行分析
    └── generate_report.py  # 生成报告
```

## 核心功能

### 1. 数据预处理
- 时间字段标准化和特征提取
- 地理坐标清洗和异常值检测
- 数据质量评估和缺失值处理
- 数据采样和分区策略

### 2. 探索性数据分析 (EDA)
- 时间模式分析（小时、日、周、月趋势）
- 空间分布热力图和聚类分析
- 业务指标统计和异常检测
- 交互式可视化仪表盘

### 3. 特征工程
- 时间特征：小时、工作日、节假日、季节
- 空间特征：地理聚类、距离特征、区域编码
- 业务特征：高峰时段、需求预测指标
- 外部特征：天气、事件、经济指标

### 4. 机器学习模型
- **聚类分析**：K-means、DBSCAN热点区域识别
- **时间序列预测**：ARIMA、Prophet、LSTM
- **需求预测**：随机森林、XGBoost、LightGBM
- **异常检测**：Isolation Forest、One-Class SVM

### 5. 高级分析
- 动态热点检测和迁移分析
- 供需平衡优化建议
- 定价策略分析和优化
- 用户行为路径分析

### 6. 交互式仪表盘 🆕
- **实时数据可视化**: 基于Streamlit的交互式仪表盘
- **多维度分析**: 时间、空间、业务洞察一体化展示
- **智能筛选**: 支持时间、区域、小时等多维度数据筛选
- **需求预测**: 参数化预测接口和可视化结果
- **实时监控**: 模拟实时数据监控界面

## 技术栈

- **数据处理**: pandas, numpy, scipy
- **机器学习**: scikit-learn, xgboost, lightgbm
- **深度学习**: tensorflow, pytorch
- **时间序列**: statsmodels, prophet
- **可视化**: matplotlib, seaborn, plotly, folium
- **地理分析**: geopandas, shapely
- **数据库**: sqlite, postgresql
- **部署**: docker, streamlit, dash
- **仪表盘**: streamlit, streamlit-folium, altair

## 安装和使用

### 环境要求
- Python 3.8+
- 8GB+ RAM
- 10GB+ 磁盘空间

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd didi
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 运行分析
```bash
python scripts/run_analysis.py
```

5. 启动交互式仪表盘 🆕
```bash
# 方法1: 使用便捷脚本
python run_dashboard.py

# 方法2: 直接使用streamlit
streamlit run src/visualization/dashboard.py
```

访问仪表盘: http://localhost:8501

## 使用示例

### 基础数据分析
```python
from src.data.data_loader import DataLoader
from src.analysis.eda import ExploratoryDataAnalysis

# 加载数据
loader = DataLoader()
df = loader.load_uber_data()

# 探索性分析
eda = ExploratoryDataAnalysis(df)
eda.generate_summary_report()
```

### 空间聚类分析
```python
from src.models.clustering import SpatialClustering

# 空间聚类
clustering = SpatialClustering()
clusters = clustering.kmeans_clustering(df, n_clusters=8)
clustering.visualize_clusters()
```

### 需求预测
```python
from src.models.prediction import DemandPredictor

# 需求预测
predictor = DemandPredictor()
model = predictor.train_xgboost_model(X_train, y_train)
predictions = predictor.predict_demand(X_test)
```

### 交互式仪表盘 🆕
```python
from src.visualization.dashboard import InteractiveDashboard

# 启动仪表盘
dashboard = InteractiveDashboard()
dashboard.run_dashboard()
```

或者直接运行：
```bash
python run_dashboard.py
```

## 测试验证

### 运行测试
```bash
# 测试仪表盘功能
python test_dashboard.py

# 测试数据加载
python test_data_loading.py

# 测试模型功能（简化版）
python tests/test_models_simple.py

# 测试模型功能（完整版）
python tests/test_models.py

# 运行所有测试
python -m pytest tests/
```

## 项目特色

1. **模块化设计**: 清晰的代码结构，易于维护和扩展
2. **配置驱动**: 通过配置文件管理参数和设置
3. **自动化流程**: 一键运行完整分析流程
4. **可视化丰富**: 多种图表类型和交互式仪表盘
5. **模型评估**: 完整的模型性能评估和对比
6. **文档完善**: 详细的代码注释和API文档
7. **交互式仪表盘**: 基于Streamlit的现代化数据可视化界面

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目链接: [https://github.com/yourusername/didi]

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 基础数据分析功能
- 空间聚类和时间序列预测
- 交互式可视化仪表盘 
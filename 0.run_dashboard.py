#!/usr/bin/env python3
"""
运行滴滴出行数据分析仪表盘
"""

import subprocess
import sys
import os

def run_dashboard():
    """运行Streamlit仪表盘"""
    try:
        # 检查是否安装了streamlit
        import streamlit
        print("✅ Streamlit已安装")
    except ImportError:
        print("❌ Streamlit未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "streamlit-folium", "altair"])
        print("✅ 安装完成")
    
    # 运行仪表盘
    print("🚀 启动仪表盘...")
    print("📊 仪表盘将在浏览器中打开: http://localhost:8501")
    print("⏹️  按 Ctrl+C 停止服务器")
    
    # 运行streamlit命令
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/visualization/dashboard.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])

if __name__ == "__main__":
    run_dashboard() 
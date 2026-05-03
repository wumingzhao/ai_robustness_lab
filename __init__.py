"""AI算法脆弱性可视化探究沙盒 - 交互式教学工具"""

import sys
import os
from pathlib import Path
import streamlit.web.cli as stcli


def main():
    """标准入口函数：供 bat脚本 / pip install / python -m 三种方式调用"""
    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), "--server.port=8501"]
    stcli.main()

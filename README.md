
<div align="center"> 

# **AI分类算法原理可视化探究沙盒**

*《从“看走眼”到“画错线”：AI分类算法的脆弱性可视化探究》*

**面向结构化与非结构化数据的决策机制剖析工具**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airobustnesslab-qpyhrdy4d6ndzea6zrniyy.streamlit.app/)
[![Download](https://img.shields.io/badge/下载-便携版-blue)](https://github.com/wumingzhao/ai_robustness_lab/releases/download/v1.0.0/ai_robustness_lab_win_portable_v1.0.zip) 
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div> 

---

> 一个将抽象算法黑盒转化为直观“空间碰撞博弈”的交互式教学工具。借助生成式AI辅助开发，让中学生也能像科学家一样探究AI脆弱性的本质。

## 🌟 核心亮点：为什么这个沙盒与众不同？

传统的AI教学工具往往只是“调参看准确率”的单向黑盒，本系统首创了**“双轨空间碰撞”机制**：

- 👁️ **感性特征漂移轨（看走眼）：** 滑动“模糊/遮挡”滑块，左侧被破坏的图像会瞬间转化为右侧散点图中 **“特征星星的坐标漂移”**。
- 🧠 **理性边界扭曲轨（画错线）：** 滑动“投毒/重叠”滑块，污染的训练数据会实时渲染为右侧 **“决策边界的几何扭曲”**。
- 💥 **空间碰撞触发机制：** 只有当“漂移的星星”刚好触碰上“扭曲的边界”时，才会触发极具视觉冲击力的红色印章掉落动效，深刻揭示系统脆弱性的复合灾难效应。

## 🛠️ 技术栈与架构

本项目采用 **“AI辅助开发 + 经典AI算法运行”** 的双线技术架构：
- **前端交互：** `Streamlit` (天生契合“调参-观察”的教学闭环)
- **算法引擎：** `Scikit-learn` (调用真实的SVM/KNN/决策树进行网格采样与边界计算)
- **图像降质：** `Pillow` (像素级高斯模糊与局部遮挡模拟)
- **高维降维：** `Matplotlib` + `NumPy` (高精度等高线图与散点漂移动画)
- **零代码解耦：** 底层逻辑与业务场景通过 `config.json` 彻底剥离。

## 🚀 快速开始

### 方式一：开箱即用（推荐！小白首选 💡）

如果你不想折腾环境配置，请直接下载 `Release` 中的便携版压缩包：

1. **解压**压缩包到任意文件夹（路径中**不要有中文**）
2. 双击运行 📄 `一键启动.bat`
3. 浏览器会自动弹出，开始探究！

> ⚠️ 注意：便携版仅限 Windows 系统使用。解压后请勿移动内部文件位置。

---

### 方式二：本地源码部署（开发者）

#### 1. 克隆项目

```bash  
git clone https://github.com/wumingzhao/ai_robustness_lab
cd ai_robustness_lab
```
#### 2. 安装依赖
```bash  
pip install -r requirements.txt
```

#### 3. 运行沙盒
```bash  
streamlit run app.py
```
## 🎨 零代码场景迁移（教师福音）

你不需要懂任何 Python 代码，只需修改 `config.json`，即可将“广彩质检”一键迁移至任何你想教学的真实场景！

由于项目采用解耦架构，你需要在 `config.json` 中配置你的图片路径，并将对应图片放入项目中：

- **示例图片 (`sample_img`):** 建议使用 4:3 比例的清晰图片（如广彩瓷器、医疗影像等）。
- **红章图片 (`seal_img`):** 建议使用带透明通道的 PNG 格式“误判”印章图片。
- **中文字体:** 请确保 `fonts/SimHei.ttf` 存在，防止图表中文乱码。

**示例：将广彩瓷器改为医疗X光片分析**
```json  
{  
“scene_name”: “医疗X光片良恶性分析”,  
“sample_img”: “assets/x_ray.jpg”,  
“seal_img”: “assets/seal_misdiagnosis.png”,  
“positive_label”: “良性”,  
“negative_label”: “恶性”,  
“feature_x”: “阴影边缘锐度”,  
“feature_y”: “病灶纹理复杂度”  
}
```
修改保存后，刷新网页，所有的UI文案、坐标轴名称、判定结果将全自动切换！

## 📁 项目目录结构


```text  
.  
├── app.py # 主程序入口  
├── config.json # 🔧 零代码场景配置文件 (修改此文件切换场景)  
├── requirements.txt # Python依赖清单  
├── fonts/  
│ └── SimHei.ttf # 中文字体文件 (防止Matplotlib乱码)  
└── assets/ # 静态资源文件夹 (需自行准备)  
├── sample.jpg # 默认的送检样本图  
└── seal.png # 默认的误判红章图 (建议透明底)
```

## 📜 教学应用价值

1. **破除AI迷信：** 课前课后问卷对比证实，能强力击碎学生“100%准确率=绝对安全”的认知偏差。
2. **落实白盒化：** 将高维黑盒降维为X/Y二维的“物理空间博弈”，让初中生也能看懂算法底线。
3. **学用结合：** 内置“探究实验台”，学生可一键快照生成CSV格式的实验日志，直接作为科学探究报告的原始数据。

## 🤝 贡献指南

欢迎一线教师与开发者提出改进建议！如果你开发了新的 `config.json` 场景（如：农作物病虫害识别、工业缺陷检测等），非常欢迎提交 Pull Request 共享到 `templates/` 目录！

## 📄 开源协议

本项目基于 [MIT](LICENSE) 协议开源。AI赋能教育，期待你的探索与传播！

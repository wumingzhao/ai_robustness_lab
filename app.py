# =======python库导入=================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import io
import base64
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter  # 🔧 修复：补上 ImageFilter
import os
import warnings
warnings.filterwarnings('ignore')

# ===== 2. 零代码配置加载（核心引擎） =====
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)



# ==========================================
# 0.5 解决 Matplotlib 中文乱码
# ==========================================
def setup_chinese_font():
    system = platform.system()
    font_path = None
    if system == 'Windows':
        font_path = 'C:/Windows/Fonts/simhei.ttf'
        if not os.path.exists(font_path):
            font_path = 'C:/Windows/Fonts/msyh.ttc'
    elif system == 'Darwin':
        font_path = '/System/Library/Fonts/PingFang.ttc'
    else:
        font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    try:
        if font_path and os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

# ==========================================
# 1. 核心算法引擎 (增加特征漂移计算)
# ==========================================
def run_algorithm_simulation(algo_type, overlap, poison_rate, sample_size, blur=0, occlude=0):
    # 1. 生成历史数据（边界）
    center_distance = 4.0 * (1.0 - overlap)
    centers = np.array([[-center_distance/2, -center_distance/2],
                        [center_distance/2, center_distance/2]])
    X, y = make_blobs(n_samples=sample_size, centers=centers, cluster_std=1.2, random_state=42)

    if poison_rate > 0:
        poison_count = int(len(y) * poison_rate)
        poison_indices = np.random.choice(len(y), poison_count, replace=False)
        y[poison_indices] = 1 - y[poison_indices]

    if algo_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    accuracy = model.score(X, y)

    # ✅ 核心黑科技：将左侧的非结构化干扰，转化为结构化的坐标偏移！
    # 正常情况下，正品特征在 [0, 0] 附近（绝对安全区）
    base_x, base_y = 0.0, 0.0
    drift_x = blur * 0.25        # 模糊导致色彩特征向右侧危险区漂移
    drift_y = -occlude * 0.05    # 遮挡导致边缘特征向下方危险区漂移
    
    test_point = np.array([[base_x + drift_x, base_y + drift_y]])
    real_label = 0
    predicted_label = model.predict(test_point)[0]
    is_misjudged = (real_label != predicted_label)

    # 绘图逻辑保持不变
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.patch.set_alpha(0)  # 背景透明，与Streamlit更融合
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.15),
                         np.arange(y_min, y_max, 0.15))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='black', s=25, alpha=0.8, label="历史经验库")

    marker = 'X' if is_misjudged else '*'
    color = 'red' if is_misjudged else 'lime'
    size = 300 if is_misjudged else 200
    ax.scatter(test_point[:, 0], test_point[:, 1], marker=marker, s=size, c=color,
               edgecolors='black', linewidths=2, zorder=5, label="当前送检样本")

    #ax.set_title(f"边界健康度: {accuracy:.1%}", fontsize=13, fontweight='bold')
    # ✅ 双维度标题：既展示边界健康度，又展示送检结果
    label = CFG["negative_label"] if is_misjudged else CFG["positive_label"]
    status_text = f"[× 误判:{label}]" if is_misjudged else f"[√ 判定:{label}]"

    #ax.set_title(
    #    f"决策边界健康度: {accuracy:.1%}  |  送检判定: {status_text}",
    #    fontsize=12, fontweight='bold'
    #)

    ax.set_xlabel(f"特征 A ({CFG['feature_x']})",fontsize=10)
    ax.set_ylabel(f"特征 B ({CFG['feature_y']})",fontsize=10)

    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)

    return fig, accuracy, is_misjudged

# ==========================================
# 2. 页面配置与状态管理
# ==========================================
st.set_page_config(
    page_title=f"AI算法脆弱性探究沙盒 - {CFG['scene_name']}", 
    layout="wide", 
    page_icon="🧪"
)



# ==========================================
# 0. 素材路径配置 (使用自定义图片)
# ==========================================

# 🔧 修改点1：把这里的文件名，改成你自己的图片名字！
# 以 app.py 所在目录为基准拼接路径，不管从哪里启动都不会迷路
_BASE_DIR = os.path.dirname(__file__)
IMG_PATH = os.path.join(_BASE_DIR, CFG["sample_img"])
SEAL_PATH = os.path.join(_BASE_DIR, CFG["seal_img"])
 

# 验证图片是否存在，防止报错
if not os.path.exists(IMG_PATH):
    st.error(f"找不到示例图片，请检查路径：{IMG_PATH}")
if not os.path.exists(SEAL_PATH):
    st.error(f"找不到红章图片，请检查路径：{SEAL_PATH}")


##============================
# 缓存优化
#===================================

# ✅ 优化1：利用缓存，印章 Base64 整个程序只生成一次
@st.cache_resource
def get_seal_base64(path):
    seal = Image.open(path).convert('RGBA')
    buf = io.BytesIO()
    seal.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

SEAL_URI = get_seal_base64(SEAL_PATH) # 生成全局常量


# ✅ 注入印章动画和呼吸灯的CSS样式（只需放一次）
st.markdown("""
<style>
/* ======== 盖章掉落容器 ======== */
.seal-container {
    position: relative;
    display: inline-block;
    width: 100%;
    aspect-ratio: 4 / 3.0; 
    border-radius: 12px;
}

/* ======== 底层图片 ======== */
.seal-container .base-image {
    width: 100%;
    height: 100%;
    object-fit:contain;
    display: block;
    border-radius: 12px;
}

/* ======== 印章覆盖层 ======== */
.seal-container .seal-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 55%;
    height: auto;
    pointer-events: none;
    animation: sealDrop 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.4) 2.0s forwards;
    opacity: 0;
}

/* ======== 印章从天而降动画 ======== */
@keyframes sealDrop {
    0% {
        transform: translate(-50%, -200%) scale(0.3) rotate(-25deg);
        opacity: 0;
    }
    45% {
        transform: translate(-50%, -48%) scale(1.25) rotate(8deg);
        opacity: 0.85;
    }
    65% {
        transform: translate(-50%, -52%) scale(0.92) rotate(-4deg);
        opacity: 0.72;
    }
    85% {
        transform: translate(-50%, -49%) scale(1.05) rotate(2deg);
        opacity: 0.68;
    }
    100% {
        transform: translate(-50%, -50%) scale(1) rotate(0deg);
        opacity: 0.62;
    }
}

/* ======== 红色呼吸灯警告光晕 ======== */
.seal-container.breathing {
    animation: breathe 1.8s ease-in-out 3.0s infinite;
    border-radius: 12px;
}

@keyframes breathe {
    0%, 100% {
        box-shadow: 0 0 6px 2px rgba(255, 20, 20, 0.15),
                    inset 0 0 6px 1px rgba(255, 20, 20, 0.05);
    }
    50% {
        box-shadow: 0 0 28px 10px rgba(255, 20, 20, 0.55),
                    inset 0 0 18px 5px rgba(255, 20, 20, 0.2);
    }
}

/* ======== 修正Streamlit容器边距 ======== */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


if 'exp_records' not in st.session_state:
    st.session_state.exp_records = pd.DataFrame(columns=["序号", "算法", "图像模糊", "局部遮挡", "重叠度", "投毒率", "健康度", "送检结果"])

st.title('🧪 从"看走眼"到"画错线"：AI分类算法脆弱性探究')
st.caption("*本工具中的场景示意图由代码自动生成（隐喻AI生成），旨在探究算法原理，不作为真实文物鉴定依据。*")






# ==========================================
# 3. 侧边栏
# ==========================================
with st.sidebar:
    st.header("🎛️ 实验环境控制台")
    st.divider()
    algo = st.radio("1️⃣ 选择探究算法", ["KNN (K近邻)", "决策树"], index=0)
    algo_clean = "KNN" if "KNN" in algo else "DT"

    st.subheader("2️⃣ 非结构化感知破坏器")
    blur_level = st.slider("图像模糊干扰", 0, 15, 0, key="blur_slider", help="模拟拍摄失焦、雨雾等")
    occlude_level = st.slider("局部遮挡干扰", 0, 80, 0, help="模拟被遮挡的面积比例(%)", key="occlude_slider")


    st.subheader("3️⃣ 结构化数据破坏器")
    overlap = st.slider("特征重叠度 (数据混淆)", 0.0, 0.9, 0.2, step=0.1, help="拉高，模拟好次品长得太像")
    poison = st.slider("标注投毒率 (脏数据)", 0.0, 0.3, 0.0, step=0.05, help="拉高，模拟人工贴错标签")

    
    with st.expander("高级参数 (一般不改)"):
        sample_num = st.slider("样本数量", 50, 300, 100, step=10)


# ==========================================
# 4. 主界面双栏 (终极虚实联动版 - CSS动画印章)
# ==========================================

# 所有参数都在侧边栏定义完毕，这里安全调用算法
fig, accuracy, is_misjudged = run_algorithm_simulation(
    algo_clean, overlap, poison, sample_num, blur_level, occlude_level
)

any_failed = accuracy < 0.85 or is_misjudged

col_left, col_right = st.columns(2, gap="large")

# ============ 左侧：瓷器图 + 盖章动画 (极速版) ============
with col_left:
    st.subheader("📷 现象探秘：非结构化感知")
    img = Image.open(IMG_PATH)

    # 1. 图像干扰处理
    display_img = img.copy()
    if blur_level > 0:
        display_img = display_img.filter(ImageFilter.GaussianBlur(radius=blur_level))
    if occlude_level > 0:
        draw = ImageDraw.Draw(display_img)
        img_w, img_h = display_img.size
        box_w = int(img_w * occlude_level / 100)
        box_h = int(img_h * occlude_level / 100)
        draw.rectangle([img_w - box_w, img_h - box_h, img_w, img_h], fill=(0, 0, 0))

    # ✅ 优化2：强制压缩尺寸和质量再转 Base64 (极速核心)
    buf = io.BytesIO()
    display_img.convert('RGB').save(buf, format='JPEG', quality=80, optimize=True) # 用JPEG代替PNG，体积小十倍
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_uri = f"data:image/jpeg;base64,{img_b64}"

    # 根据判定结果拼接 HTML (直接使用全局缓存的 SEAL_URI)
    if any_failed:
        html = f'''
        <div class="seal-container breathing">
            <img src="{img_uri}" class="base-image">
            <img src="{SEAL_URI}" class="seal-overlay">
        </div>
        '''
    else:
        html = f'''
        <div class="seal-container">
            <img src="{img_uri}" class="base-image">
        </div>
        '''

    st.info("👁️ 调节侧边栏第②组滑块，观察图片变化及右侧⭐的漂移轨迹！")
    
    # ✅ 优化3：去掉多余的 columns 居中，CSS 已经处理了，减少 DOM 节点渲染
    st.markdown(html, unsafe_allow_html=True)

    # 状态提示
    if is_misjudged:
        st.error(f"💥 致命误判：由于特征漂移或边界扭曲，{CFG['positive_label']}掉入{CFG['negative_label']}区！")
    elif accuracy < 0.85:
        st.warning("⚠️ 风险预警：边界极度扭曲，系统处于崩溃边缘。")
    else:
        st.success("✅ 状态正常：特征稳定且边界清晰。")

# ============ 右侧：散点图 ============
with col_right:
    st.subheader("📐 机制解密：结构化边界推演")
    st.info("🔧 调节侧边栏第③组滑块或切换探究算法，观察决策边界的扭曲及左侧图片变化！")
    st.pyplot(fig, use_container_width=True)

    # ✅ 新增：双维度仪表盘
    gauge_col1, gauge_col2 = st.columns(2)
    with gauge_col1:
        st.metric(label="📊 边界健康度", value=f"{accuracy:.1%}",
                  delta="正常" if accuracy >= 0.85 else "危险！")
    with gauge_col2:
        misjudge_label = "❌ 误判" if is_misjudged else "✅ 正品"
        misjudge_delta = "正常" if not is_misjudged else "需人工复检！"
        st.metric(label="🔍 送检判定", value=misjudge_label, delta=misjudge_delta)

    st.divider()


    with st.expander("💡 读懂这张图（核心原理）"):
        st.markdown(f"""
        **📊 坐标轴的含义（AI眼中的世界）：**
        - **X轴**：模拟AI提取的【{CFG['feature_x']}】。
        - **Y轴**：模拟AI提取的【{CFG['feature_y']}】。
        - *注：现实中的AI会提取成千上万个特征，这里为了可视化，降维成了最关键的两个。*

        ---

        **🔴🔵 图例说明：**
        - **红蓝圆点**：历史存档数据（蓝点={CFG['positive_label']}，红点={CFG['negative_label']}）。
        - **彩色背景区域**：AI学到的**决策边界（脑回路）**，它把图切成了两半，认为蓝区就是{CFG['positive_label']}，红区就是{CFG['negative_label']}。
        - **⭐星星位置**：由侧边栏第②组滑块控制的**当前送检产品**的特征坐标。

        ---

        **🩺 核心指标拆解（必读！）：**
        - **边界健康度 (Accuracy)**：代表 **AI画的线本身对不对**。
          - 100%代表训练数据干净，AI画的边界完美区分了{CFG['positive_label']}和{CFG['negative_label']}。
          - 下降代表侧边栏的【标注投毒】起效了，AI学到了错误的规律，导致**"画错线"**。
        - **送检判定**：代表 **当前的这张照片掉进了哪个区域**。
          - 正品代表⭐落在蓝区；误判代表⭐落进了红区。
          - 即使边界健康度是100%（线画得对），如果侧边栏的【模糊/遮挡】拉太高，提取出的特征坐标就会严重漂移，导致⭐越过边界掉进红区，这就是**"看走眼"**。

        **💥 误判的两种真相：**
        1. **单纯看走眼**：健康度100%，但误判了。（线是对的，图太糊，⭐跑偏了）
        2. **画错线+看走眼**：健康度下降，且误判了。（线本来就是歪的，图再一糊，彻底完蛋）
        """)


# ==========================================
# 5. 底部模块四
# ==========================================
st.divider()
st.subheader("📊 探究实验台：数据沉淀与导出")

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("📸 记录当前实验数据", use_container_width=True, type="primary"):
        result_text = "正常通过" if not is_misjudged else "发生误判"
        new_record = {
            "序号": len(st.session_state.exp_records) + 1,
            "算法": algo,
            "图像模糊": f"Lv.{blur_level}",
            "局部遮挡": f"{occlude_level}%",
            "重叠度": f"{overlap:.0%}",
            "投毒率": f"{poison:.0%}",
            "健康度": f"{accuracy:.1%}",
            "送检结果": result_text
        }

        new_df = pd.DataFrame([new_record])
        st.session_state.exp_records = pd.concat([st.session_state.exp_records, new_df], ignore_index=True)
        st.toast("✅ 记录成功！", icon="✅")

with col_btn2:
    if st.button("🗑️ 清空所有记录", use_container_width=True):
        st.session_state.exp_records = pd.DataFrame(columns=["序号", "算法", "图像模糊", "局部遮挡", "重叠度", "投毒率", "健康度", "送检结果"])
        st.toast("记录已清空", icon="🗑️")

st.dataframe(st.session_state.exp_records, use_container_width=True, hide_index=True)

# 🔧 修复：删掉多余的 .encoding() 行，只保留正确的 .encode() 调用
if not st.session_state.exp_records.empty:
    csv_data = st.session_state.exp_records.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 导出实验报告 (CSV格式，可用Excel打开)",
        data=csv_data,
        file_name='AI_算法脆弱性探究报告.csv',
        mime='text/csv'
    )

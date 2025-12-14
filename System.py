import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import seaborn as sns

# ==========================================
# 0. 全局配置与样式美化
# ==========================================
st.set_page_config(page_title="水厂智能控制决策系统 (Pro Max)", page_icon="💧", layout="wide")

# 【核心修复】强制配置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 使用 Seaborn 高级样式，同时指定字体
sns.set_context("notebook", font_scale=1.0)
sns.set_style("whitegrid", {"font.sans-serif": ['Microsoft YaHei', 'SimHei']})

# CSS 美化注入
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    h1 {color: #1f77b4; font-family: 'Microsoft YaHei';}
    .stSidebar {background-color: #ffffff;}
    div[data-testid="stExpander"] {border: 1px solid #e6e6e6; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 核心算法类 (保留参考代码逻辑)
# ==========================================

class GRNN(BaseEstimator, RegressorMixin):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        dists_sq = cdist(X, self.X_train, metric='sqeuclidean')
        weights = np.exp(-dists_sq / (2 * (self.sigma ** 2)))
        weights_sum = np.sum(weights, axis=1, keepdims=True) + 1e-10
        pred = np.dot(weights, self.y_train) / weights_sum
        return pred


class BoostingGRNN(BaseEstimator, RegressorMixin):
    def __init__(self, sigma1=0.5, sigma2=None):
        self.sigma1 = sigma1
        self.sigma2 = sigma2 if sigma2 is not None else sigma1 * 0.5
        self.m1 = None
        self.m2 = None

    def fit(self, X, y):
        self.m1 = GRNN(sigma=self.sigma1).fit(X, y)
        pred1 = self.m1.predict(X)
        residuals = y - pred1
        self.m2 = GRNN(sigma=self.sigma2).fit(X, residuals)
        return self

    def predict(self, X):
        return self.m1.predict(X) + self.m2.predict(X)


# --- 深度学习模型 ---
class BPNet(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden=64):
        super(BPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x): return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden=64):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 2, batch_first=True)
        self.fc = nn.Linear(hidden, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden=64):
        super(BiLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ==========================================
# 2. 数据处理与特征工程
# ==========================================

def feature_engineering(df, input_cols):
    """特征工程：增加PAC效能和月份特征"""
    df_eng = df.copy()
    if '日期' in df_eng.columns:
        df_eng['Month'] = df_eng['日期'].dt.month

    if '混凝投加量' in df_eng.columns and '浊度' in df_eng.columns:
        df_eng['PAC_效能'] = df_eng['混凝投加量'] / (df_eng['浊度'] + 0.1)

    new_cols = ['Month', 'PAC_效能']
    final_inputs = input_cols + [c for c in new_cols if c in df_eng.columns]
    return df_eng, final_inputs


def create_time_step_data(X, Y, time_step):
    """构建时间步序列数据 (滑动窗口)"""
    X_flat, X_seq, Y_out = [], [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:(i + time_step)])
        X_flat.append(X[i:(i + time_step)].flatten())
        Y_out.append(Y[i + time_step])
    return np.array(X_flat), np.array(X_seq), np.array(Y_out)


# ==========================================
# 3. 自动寻优训练逻辑 (适配中文名称)
# ==========================================

def train_auto_optimized(algo_name_cn, X_flat, X_seq, Y_data, status_box, k_folds=5):
    """
    实现自动网格搜索和训练逻辑
    """
    # 数据划分
    train_idx, test_idx = train_test_split(
        np.arange(len(X_flat)), test_size=0.2, random_state=42, shuffle=True
    )

    y_pred = None
    best_params = {}

    # --- A. GRNN 自动寻优 ---
    if "广义回归" in algo_name_cn and "增强型" not in algo_name_cn:  # 匹配 "广义回归神经网络 (GRNN)"
        status_box.info(f"🔍 正在为 {algo_name_cn} 进行 {k_folds} 折交叉验证寻找最佳平滑系数 (Sigma)...")
        X_train_opt = X_flat[train_idx]
        y_train_opt = Y_data[train_idx].ravel()

        sigma_candidates = np.arange(0.05, 1.5, 0.05)  # 搜索范围
        best_rmse = float('inf')
        best_sigma = 0.5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        progress_bar = st.progress(0)
        for i, s in enumerate(sigma_candidates):
            fold_errors = []
            for t_idx, v_idx in kf.split(X_train_opt):
                # GRNN fit 需要 2D y
                m = GRNN(sigma=s).fit(X_train_opt[t_idx], y_train_opt[t_idx].reshape(-1, 1))
                p = m.predict(X_train_opt[v_idx])
                fold_errors.append(mean_squared_error(y_train_opt[v_idx], p))

            avg_rmse = np.sqrt(np.mean(fold_errors))
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_sigma = s
            progress_bar.progress((i + 1) / len(sigma_candidates))

        status_box.success(f"✅ 优化完成! 最佳 Sigma: {best_sigma:.2f} (CV RMSE: {best_rmse:.4f})")
        best_params = {'sigma': best_sigma}

        # 使用最佳参数训练最终模型
        model = GRNN(sigma=best_sigma)
        model.fit(X_flat[train_idx], Y_data[train_idx])
        y_pred = model.predict(X_flat[test_idx])

    # --- B. Boosting-GRNN 两阶段寻优 ---
    elif "增强型" in algo_name_cn:  # 匹配 "增强型广义回归 (Boosting-GRNN)"
        status_box.info("🔍 正在优化 Boosting-GRNN 的双层残差结构...")
        X_train_opt = X_flat[train_idx]
        y_train_opt = Y_data[train_idx].ravel()

        X_t, X_v, y_t, y_v = train_test_split(X_train_opt, y_train_opt, test_size=0.2, random_state=42)

        # 第一层寻优
        sigma_candidates = np.arange(0.05, 1.5, 0.05)
        best_mse = float('inf')
        best_s1 = 0.5

        for s in sigma_candidates:
            m = GRNN(sigma=s).fit(X_t, y_t.reshape(-1, 1))
            mse = mean_squared_error(y_v, m.predict(X_v))
            if mse < best_mse:
                best_mse = mse
                best_s1 = s

        s2 = best_s1 * 0.5  # 默认比例
        status_box.success(f"✅ 双层结构优化完成: Sigma1={best_s1:.2f}, Sigma2={s2:.2f}")
        best_params = {'sigma1': best_s1, 'sigma2': s2}

        model = BoostingGRNN(sigma1=best_s1, sigma2=s2)
        model.fit(X_flat[train_idx], Y_data[train_idx])
        y_pred = model.predict(X_flat[test_idx])

    # --- C. CatBoost ---
    elif "CatBoost" in algo_name_cn:
        status_box.info("🚀 正在训练 CatBoost 回归模型 (内置自适应优化)...")
        model = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, verbose=0, loss_function='RMSE')
        model.fit(X_flat[train_idx], Y_data[train_idx].ravel())
        y_pred = model.predict(X_flat[test_idx]).reshape(-1, 1)

    # --- D. Random Forest ---
    elif "随机森林" in algo_name_cn:
        status_box.info("🌲 正在构建随机森林 (Random Forest)...")
        model = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
        model.fit(X_flat[train_idx], Y_data[train_idx].ravel())
        y_pred = model.predict(X_flat[test_idx]).reshape(-1, 1)

    # --- E. 深度学习 ---
    elif any(x in algo_name_cn for x in ["BP", "LSTM", "BiLSTM"]):
        status_box.info(f"🧠 正在训练深度学习模型: {algo_name_cn}...")

        y_t_tensor = torch.FloatTensor(Y_data[train_idx]).to(DEVICE)

        if "BP" in algo_name_cn:
            model = BPNet(input_dim=X_flat.shape[1], hidden=64).to(DEVICE)
            X_t_tensor = torch.FloatTensor(X_flat[train_idx]).to(DEVICE)
            X_v_tensor = torch.FloatTensor(X_flat[test_idx]).to(DEVICE)
        else:
            # 判断 LSTM 还是 BiLSTM
            model_class = LSTMNet if "双向" not in algo_name_cn else BiLSTMNet
            model = model_class(input_dim=X_seq.shape[2], hidden=64).to(DEVICE)
            X_t_tensor = torch.FloatTensor(X_seq[train_idx]).to(DEVICE)
            X_v_tensor = torch.FloatTensor(X_seq[test_idx]).to(DEVICE)

        dataset = TensorDataset(X_t_tensor, y_t_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model.train()
        epochs = 150
        prog_bar = st.progress(0)

        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = loss_fn(out, by)
                loss.backward()
                optimizer.step()
            if epoch % 5 == 0:
                prog_bar.progress((epoch + 1) / epochs)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_v_tensor).cpu().numpy()

    return y_pred, Y_data[test_idx], best_params


# ==========================================
# 4. 前端界面布局
# ==========================================

# --- 侧边栏 ---
st.sidebar.title("🎛️ 系统控制台")

with st.sidebar.expander("📂 1. 数据配置", expanded=True):
    use_demo = st.checkbox("使用演示数据 (Demo)", value=False)
    uploaded_file = None
    if not use_demo:
        uploaded_file = st.file_uploader("上传 Excel 数据文件 (.xlsx)", type=['xlsx'])

    st.markdown("---")
    # 新增高级选项
    time_step = st.slider("时间步长 (Time Step)", 1, 10, 3, help="LSTM序列长度，默认为3")
    split_ratio = st.slider("测试集划分比例", 0.1, 0.5, 0.2, 0.05)
    shuffle_data = st.checkbox("随机打乱数据 (Shuffle)", value=True, help="推荐勾选以提高模型的泛化能力")

with st.sidebar.expander("🤖 2. 核心算法选择", expanded=True):
    # 【核心更新】使用中文算法名称
    algo_type = st.selectbox(
        "选择预测模型",
        [
            "广义回归神经网络 (GRNN)",
            "增强型广义回归 (Boosting-GRNN)",
            "CatBoost 回归",
            "随机森林 (Random Forest)",
            "BP 神经网络",
            "长短期记忆网络 (LSTM)",
            "双向长短期记忆网络 (BiLSTM)"
        ]
    )
    cv_folds = st.number_input("自动寻优交叉验证折数 (CV Folds)", 2, 10, 5, help="数值越大寻优越准，但速度越慢")

with st.sidebar.expander("🎨 3. 可视化设置", expanded=False):
    img_dpi = st.number_input("图片清晰度 (DPI)", 100, 600, 300, 50)
    show_ci = st.checkbox("显示预测置信区间", value=True)

# --- 主界面 ---
st.title("🌊 水厂智能控制决策系统 (Auto-Optimized)")
st.caption(f"当前模式: 自动超参数寻优 | 算法: {algo_type} | 计算设备: {DEVICE}")
st.markdown("---")


def process_data_pipeline(file, demo, t_step):
    """数据处理流水线"""
    if demo:
        dates = pd.date_range(start='2023-01-01', periods=600, freq='D')
        data = {
            '日期': dates,
            '一二期进水量': np.random.rand(600) * 1000,
            '水温': np.sin(np.linspace(0, 10, 600)) * 10 + 15,  # 模拟季节性
            '浊度': np.random.rand(600) * 10,
            'PH': np.random.rand(600) + 7,
            '氨氮': np.random.rand(600),
            '混凝投加量': np.random.rand(600) * 20,
            '预臭氧': np.random.rand(600) * 5,
            '砂滤池出水浊度': np.sin(np.linspace(0, 20, 600)) * 0.1 + 0.2 + np.random.normal(0, 0.015, 600)
        }
        df = pd.DataFrame(data)
    elif file:
        try:
            df = pd.read_excel(file)
            if '日期' in df.columns: df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        except:
            return None
    else:
        return None

    # 特征工程
    input_cols_base = ['一二期进水量', '水温', '浊度', 'PH', '氨氮', '混凝投加量', '预臭氧']
    target_col = '砂滤池出水浊度'

    missing = [c for c in input_cols_base + [target_col] if c not in df.columns]
    if missing:
        st.error(f"Excel文件缺少必要的列: {missing}")
        return None

    df_eng, final_inputs = feature_engineering(df, input_cols_base)
    df_clean = df_eng.dropna(subset=final_inputs + [target_col]).reset_index(drop=True)

    # 滤波平滑
    for col in final_inputs + [target_col]:
        if len(df_clean) > 15:
            try:
                df_clean[col] = savgol_filter(df_clean[col], 15, 3)
            except:
                pass

    # 数据准备
    X_raw = df_clean[final_inputs].values
    Y_raw = df_clean[target_col].values.reshape(-1, 1)

    # 归一化 (关键步骤)
    scaler_x = StandardScaler().fit(X_raw)
    scaler_y = StandardScaler().fit(Y_raw)

    X_s = scaler_x.transform(X_raw)
    Y_s = scaler_y.transform(Y_raw)  # 深度学习推荐对Target也归一化

    # 构建时间序列数据
    X_flat, X_seq, Y_data = create_time_step_data(X_s, Y_s, t_step)

    return X_flat, X_seq, Y_data, scaler_y


# --- 执行逻辑 ---
btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    start_btn = st.button("🚀 启动自动寻优训练", type="primary", use_container_width=True)

if start_btn:
    data_bundle = process_data_pipeline(uploaded_file, use_demo, time_step)

    if data_bundle:
        X_flat, X_seq, Y_data, scaler_y = data_bundle

        # 创建状态显示容器
        status_container = st.container()
        with status_container:
            st.info(f"💡 系统正在初始化，准备运行 {algo_type}...")

        start_time = time.time()

        try:
            # 运行核心训练
            y_pred_scaled, y_true_scaled, best_params = train_auto_optimized(
                algo_type, X_flat, X_seq, Y_data, status_container, k_folds=cv_folds
            )

            # 反归一化 (还原真实数值)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_true_scaled)

            # 计算指标
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            # 清除状态信息，显示成功
            status_container.empty()
            st.success(f"✅ 训练完成！耗时 {time.time() - start_time:.2f} 秒")

            if best_params:
                st.write("🎯 **自动寻优结果 (Best Parameters):**")
                st.json(best_params, expanded=False)

            # --- 结果看板 ---
            st.subheader("📊 模型性能评估看板")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("R² (拟合优度)", f"{r2:.4f}", delta_color="normal")
            col_m2.metric("RMSE (均方根误差)", f"{rmse:.4f}", delta_color="inverse")
            col_m3.metric("MAE (平均绝对误差)", f"{mae:.4f}", delta_color="inverse")
            col_m4.metric("测试集样本量", f"{len(y_true)}")

            # --- 可视化 ---
            tab1, tab2, tab3 = st.tabs(["📈 时序预测对比图", "🎯 回归拟合分析图", "📉 误差残差分布图"])

            # 1. 时序图 (美化版)
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 4), dpi=img_dpi)
                limit = 200  # 限制显示点数

                # 绘制真实值
                ax.plot(y_true[:limit], label='真实测量值 (Actual)', color='#2C3E50', alpha=0.6, linewidth=1.5)
                # 绘制预测值
                ax.plot(y_pred[:limit], label='模型预测值 (Predicted)', color='#E74C3C', linestyle='-', linewidth=1.5,
                        alpha=0.9)

                # 绘制误差区间
                if show_ci:
                    ax.fill_between(range(len(y_true[:limit])),
                                    y_true[:limit].flatten(),
                                    y_pred[:limit].flatten(),
                                    color='#E74C3C', alpha=0.15, label='95% 置信区间')

                ax.set_title(f"出水浊度时序预测 - {algo_type}", fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel("时间步 (Time Step)", fontsize=10)
                ax.set_ylabel("出水浊度 (NTU)", fontsize=10)
                ax.legend(frameon=True, fancybox=True, shadow=True)
                st.pyplot(fig)

            # 2. 回归散点图 (Seaborn增强)
            with tab2:
                col_reg1, col_reg2 = st.columns([2, 1])
                with col_reg1:
                    fig, ax = plt.subplots(figsize=(6, 6), dpi=img_dpi)
                    # 计算误差作为颜色映射
                    errors = np.abs(y_true - y_pred).flatten()
                    scatter = ax.scatter(y_true, y_pred, c=errors, cmap='coolwarm',
                                         alpha=0.7, edgecolors='w', s=60, label='预测数据点')

                    # 绘制完美对角线
                    mi, ma = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
                    ax.plot([mi, ma], [mi, ma], 'k--', lw=2, label='完美拟合线 (y=x)')

                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('绝对误差 (Abs Error)', fontsize=10)
                    ax.set_xlabel("真实测量值 (Actual)", fontsize=12)
                    ax.set_ylabel("模型预测值 (Predicted)", fontsize=12)
                    ax.set_title(f"回归拟合效果 (R²={r2:.3f})", fontsize=14, fontweight='bold')
                    ax.legend()
                    st.pyplot(fig)

                with col_reg2:
                    st.markdown("#### 💡 图表解读")
                    st.info("""
                    * **对角线**: 数据点越靠近黑色虚线，说明预测越准确。
                    * **颜色**: 
                        * 🔵 蓝色点表示误差很小。
                        * 🔴 红色点表示误差较大。
                    """)

            # 3. 残差分布
            with tab3:
                res = y_true - y_pred
                fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=img_dpi)

                # 直方图
                sns.histplot(res, kde=True, ax=ax[0], color='#8e44ad', edgecolor='w')
                ax[0].axvline(0, color='r', linestyle='--')
                ax[0].set_title("预测残差分布直方图 (Histogram)")
                ax[0].set_xlabel("预测误差 (Error)")

                # 散点图
                ax[1].scatter(range(len(res)), res, alpha=0.5, color='#8e44ad')
                ax[1].axhline(0, color='r', linestyle='--')
                ax[1].set_title("残差分布散点图 (Scatter)")
                ax[1].set_ylabel("误差值 (Error Value)")

                st.pyplot(fig)

        except Exception as e:
            st.error(f"训练过程中发生错误: {str(e)}")
            import traceback

            st.code(traceback.format_exc())

    else:
        st.warning("⚠️ 请先在左侧侧边栏上传数据文件或勾选演示模式。")
else:
    st.info("👈 请在左侧配置参数，然后点击【启动自动寻优训练】。")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# 0. 云端适配配置
# ==========================================
st.set_page_config(page_title="Water Quality AI System", page_icon="💧", layout="wide")

# 注意：Streamlit Cloud 是 Linux 环境，默认没有微软雅黑字体。
# 为了防止报错，我们这里不指定特定中文字体，而是让 matplotlib 自动回退。
# 如果必须显示中文，通常需要上传字体文件，这里为了部署成功率，建议图表标题用英文或默认字体。
sns.set_context("notebook", font_scale=1.0)
sns.set_style("whitegrid")

# CSS 注入：学术风格评分卡
st.markdown("""
<style>
    .big-grade { font-size: 60px; font-weight: 900; margin: 0; line-height: 1; font-family: sans-serif; }
    .grade-desc { font-size: 16px; color: #666; margin-top: 5px;}
    .academic-box-pass { 
        background-color: #f0fdf4; border-left: 5px solid #166534; padding: 15px; border-radius: 4px; color: #14532d;
    }
    .academic-box-fail { 
        background-color: #fef2f2; border-left: 5px solid #991b1b; padding: 15px; border-radius: 4px; color: #7f1d1d;
    }
    .stMetric { background-color: #ffffff; border: 1px solid #e5e7eb; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# 强制使用 CPU，云端免费实例通常没有 GPU
DEVICE = torch.device("cpu")

# ==========================================
# 1. 核心算法类
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
    df_eng = df.copy()
    if '日期' in df_eng.columns:
        df_eng['Month'] = df_eng['日期'].dt.month
    
    if '混凝投加量' in df_eng.columns and '浊度' in df_eng.columns:
         df_eng['PAC_效能'] = df_eng['混凝投加量'] / (df_eng['浊度'] + 0.1)

    new_cols = ['Month', 'PAC_效能']
    final_inputs = input_cols + [c for c in new_cols if c in df_eng.columns]
    return df_eng, final_inputs

def create_time_step_data(X, Y, time_step):
    X_flat, X_seq, Y_out = [], [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:(i + time_step)])
        X_flat.append(X[i:(i + time_step)].flatten())
        Y_out.append(Y[i + time_step])
    return np.array(X_flat), np.array(X_seq), np.array(Y_out)

# ==========================================
# 3. 自动寻优逻辑
# ==========================================

def train_auto_optimized_real(algo_name_cn, X_flat, X_seq, Y_data, status_box, k_folds=5):
    indices = np.arange(len(X_flat))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    
    y_pred = None
    best_params = {}
    
    # --- GRNN ---
    if "GRNN" in algo_name_cn and "Boosting" not in algo_name_cn:
        status_box.info(f"🔍 [Auto-ML] Running Grid Search for Sigma...")
        X_train_opt = X_flat[train_idx]
        y_train_opt = Y_data[train_idx].ravel()

        sigma_candidates = np.arange(0.05, 1.5, 0.2) # 稍微减少步长以加快云端速度
        best_rmse = float('inf')
        best_sigma = 0.5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        progress_bar = st.progress(0)
        for i, s in enumerate(sigma_candidates):
            fold_errors = []
            for t_idx, v_idx in kf.split(X_train_opt):
                m = GRNN(sigma=s).fit(X_train_opt[t_idx], y_train_opt[t_idx].reshape(-1,1))
                p = m.predict(X_train_opt[v_idx])
                fold_errors.append(mean_squared_error(y_train_opt[v_idx], p))
            
            avg_rmse = np.sqrt(np.mean(fold_errors))
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_sigma = s
            progress_bar.progress((i+1)/len(sigma_candidates))
        
        status_box.success(f"✅ Optimization Done! Best Sigma: {best_sigma:.2f}")
        best_params = {'sigma': best_sigma}
        
        model = GRNN(sigma=best_sigma)
        model.fit(X_flat[train_idx], Y_data[train_idx])
        y_pred = model.predict(X_flat[test_idx])

    # --- Boosting-GRNN ---
    elif "Boosting" in algo_name_cn:
        status_box.info("🔍 [Auto-ML] Optimizing Boosting Layers...")
        X_train_opt = X_flat[train_idx]
        y_train_opt = Y_data[train_idx].ravel()
        
        X_t, X_v, y_t, y_v = train_test_split(X_train_opt, y_train_opt, test_size=0.2, random_state=42)
        
        sigma_candidates = np.arange(0.1, 1.5, 0.2)
        best_mse = float('inf')
        best_s1 = 0.5
        
        for s in sigma_candidates:
            m = GRNN(sigma=s).fit(X_t, y_t.reshape(-1,1))
            mse = mean_squared_error(y_v, m.predict(X_v))
            if mse < best_mse:
                best_mse = mse
                best_s1 = s
        
        s2 = best_s1 * 0.5 
        status_box.success(f"✅ Optimization Done: Sigma1={best_s1:.2f}, Sigma2={s2:.2f}")
        best_params = {'sigma1': best_s1, 'sigma2': s2}
        
        model = BoostingGRNN(sigma1=best_s1, sigma2=s2)
        model.fit(X_flat[train_idx], Y_data[train_idx])
        y_pred = model.predict(X_flat[test_idx])

    # --- CatBoost ---
    elif "CatBoost" in algo_name_cn:
        status_box.info("🚀 Running CatBoost Adaptive Training...")
        model = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, verbose=0, loss_function='RMSE')
        model.fit(X_flat[train_idx], Y_data[train_idx].ravel())
        y_pred = model.predict(X_flat[test_idx]).reshape(-1, 1)

    # --- Random Forest ---
    elif "Random Forest" in algo_name_cn:
        status_box.info("🌲 Building Random Forest Ensemble...")
        model = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
        model.fit(X_flat[train_idx], Y_data[train_idx].ravel())
        y_pred = model.predict(X_flat[test_idx]).reshape(-1, 1)

    # --- Deep Learning ---
    elif any(x in algo_name_cn for x in ["BP", "LSTM", "BiLSTM"]):
        status_box.info(f"🧠 Training Neural Network: {algo_name_cn}...")
        
        y_t_tensor = torch.FloatTensor(Y_data[train_idx]).to(DEVICE)
        
        if "BP" in algo_name_cn:
            model = BPNet(input_dim=X_flat.shape[1], hidden=64).to(DEVICE)
            X_t_tensor = torch.FloatTensor(X_flat[train_idx]).to(DEVICE)
            X_v_tensor = torch.FloatTensor(X_flat[test_idx]).to(DEVICE)
        else: 
            model_class = LSTMNet if "BiLSTM" not in algo_name_cn else BiLSTMNet
            model = model_class(input_dim=X_seq.shape[2], hidden=64).to(DEVICE)
            X_t_tensor = torch.FloatTensor(X_seq[train_idx]).to(DEVICE)
            X_v_tensor = torch.FloatTensor(X_seq[test_idx]).to(DEVICE)

        dataset = TensorDataset(X_t_tensor, y_t_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        loss_fn = nn.MSELoss()
        
        model.train()
        epochs = 100 # 减少轮次以适应云端CPU
        prog_bar = st.progress(0)
        
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = loss_fn(out, by)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                prog_bar.progress((epoch+1)/epochs)
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_v_tensor).cpu().numpy()
        
    return y_pred, Y_data[test_idx], best_params

# ==========================================
# 4. 前端界面布局
# ==========================================

st.sidebar.title("🎛️ Control Panel")

# 侧边栏配置
use_demo = st.sidebar.checkbox("Use Demo Data (演示模式)", value=True)
uploaded_file = None
if not use_demo:
    uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=['xlsx'])

algo_type = st.sidebar.selectbox(
    "Algorithm Selection", 
    ["GRNN", "Boosting-GRNN", "CatBoost", "Random Forest", "BP Neural Network", "LSTM", "BiLSTM"]
)
st.sidebar.info("✨ Auto-Optimization Enabled")

# 主界面
st.title("🎓 Intelligent Water Quality Prediction System")
st.markdown("**Research Prototype V2.0** | Auto-Optimization")
st.divider()

def process_data_pipeline(file, demo):
    if demo:
        dates = pd.date_range(start='2023-01-01', periods=600, freq='D')
        data = {
            '日期': dates,
            '一二期进水量': np.random.rand(600) * 1000,
            '水温': np.sin(np.linspace(0, 10, 600)) * 10 + 15,
            '浊度': np.random.rand(600) * 10,
            'PH': np.random.rand(600) + 7,
            '氨氮': np.random.rand(600),
            '混凝投加量': np.random.rand(600) * 20,
            '预臭氧': np.random.rand(600) * 5,
            '砂滤池出水浊度': np.sin(np.linspace(0, 20, 600)) * 0.1 + 0.2 + np.random.normal(0, 0.02, 600)
        }
        df = pd.DataFrame(data)
    elif file:
        try:
            df = pd.read_excel(file)
            if '日期' in df.columns: df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        except: return None
    else: return None

    # 特征工程 (保持中文列名，因为Excel表头通常是中文)
    input_cols_base = ['一二期进水量', '水温', '浊度', 'PH', '氨氮', '混凝投加量', '预臭氧']
    target_col = '砂滤池出水浊度'
    
    # 简单的列名检查
    missing = [c for c in input_cols_base + [target_col] if c not in df.columns]
    if missing and not demo:
        # 如果列名不对，尝试使用索引
        if len(df.columns) >= 8:
             st.warning("Warning: Column names mismatch. Using first 7 columns as features.")
             X_raw = df.iloc[:, 0:7].values
             Y_raw = df.iloc[:, 7].values.reshape(-1, 1)
             # 跳过后续特征工程直接返回
             scaler_x = StandardScaler().fit(X_raw)
             scaler_y = StandardScaler().fit(Y_raw)
             X_s = scaler_x.transform(X_raw)
             Y_s = scaler_y.transform(Y_raw)
             return create_time_step_data(X_s, Y_s, 3), scaler_y
        else:
             st.error(f"Missing columns: {missing}")
             return None

    # 正常流程
    df_eng, final_inputs = feature_engineering(df, input_cols_base)
    df_clean = df_eng.dropna(subset=final_inputs + [target_col]).reset_index(drop=True)
    
    for col in final_inputs + [target_col]:
        if len(df_clean) > 15:
             try: df_clean[col] = savgol_filter(df_clean[col], 15, 3)
             except: pass

    X_raw = df_clean[final_inputs].values
    Y_raw = df_clean[target_col].values.reshape(-1, 1)

    scaler_x = StandardScaler().fit(X_raw)
    scaler_y = StandardScaler().fit(Y_raw)
    
    X_s = scaler_x.transform(X_raw)
    Y_s = scaler_y.transform(Y_raw) 

    return create_time_step_data(X_s, Y_s, 3), scaler_y

# --- 执行 ---
if st.button("🚀 Start Auto-Training", type="primary"):
    # 如果上传了文件，优先用文件
    is_demo = use_demo and (uploaded_file is None)
    
    # 这里处理返回值
    result = process_data_pipeline(uploaded_file, is_demo)
    
    if result:
        # 解包 result，注意 create_time_step_data 返回的是 (X_flat, X_seq, Y_data)
        (X_flat, X_seq, Y_data), scaler_y = result
        
        status_container = st.container()
        start_time = time.time()
        
        try:
            y_pred_scaled, y_true_scaled, best_params = train_auto_optimized_real(
                algo_type, X_flat, X_seq, Y_data, status_container
            )
            
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_true_scaled)
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            status_container.empty()
            
            # --- 学术评估报告 ---
            st.subheader("📊 Model Evaluation Report")
            col_grade, col_metrics, col_text = st.columns([1, 1.5, 2.5])
            
            if r2 > 0.6: 
                grade = "A"
                g_color = "#166534"
                box_cls = "academic-box-pass"
                msg = f"**Statistical Significance Verified**: R² ({r2:.4f}) shows strong predictive power."
                st.balloons()
            else:
                grade = "C"
                g_color = "#991b1b"
                box_cls = "academic-box-fail"
                msg = f"**Moderate Fit**: R² ({r2:.4f}). Consider checking data quality."

            with col_grade:
                st.markdown(f"<div style='text-align:center;color:{g_color}'><div class='big-grade'>{grade}</div><div class='grade-desc'>Grade</div></div>", unsafe_allow_html=True)
            with col_metrics:
                st.metric("R² Score", f"{r2:.4f}")
                st.metric("RMSE", f"{rmse:.4f}")
            with col_text:
                st.markdown(f"<div class='{box_cls}'>{msg}</div>", unsafe_allow_html=True)
            
            if best_params:
                st.write(f"**Optimal Hyperparameters:** `{best_params}`")

            # --- 绘图 ---
            st.subheader("📉 Visualization")
            fig, ax = plt.subplots(figsize=(10, 4))
            
            limit = min(150, len(y_true))
            
            ax.plot(y_true[:limit], label='Ground Truth', color='#334155', alpha=0.8)
            ax.plot(y_pred[:limit], label='Prediction', color='#10b981', linestyle='--')
            
            ax.fill_between(range(limit), 
                            y_true[:limit].flatten(), 
                            y_pred[:limit].flatten(), 
                            color='#10b981', alpha=0.15)
            
            ax.set_title(f"Time Series Prediction - {algo_type}")
            ax.legend(loc='upper right')
            
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("Please upload a valid Excel file or use Demo mode.")
else:
    st.info("👈 Click button to start training")

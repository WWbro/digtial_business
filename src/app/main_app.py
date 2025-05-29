import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose # 用于时间序列分解图
from sklearn.model_selection import train_test_split # 确保正确导入
import sys


# --- 配置 Streamlit 页面 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # 获取项目根目录 (disbusiness)
sys.path.append(project_root) # 将项目根目录添加到 sys.path
sys.path.append(os.path.join(project_root, 'src')) # 将 src 目录也添加到 sys.path，便于模块查找

try:
    from src.data_processing.processor import DataProcessor
    from src.modeling.forecaster import SalesForecaster
    from src.visualization.plotter import Plotter
except ImportError as e:
    st.error(f"导入模块时出错: {e}。请确保您的 PYTHONPATH 设置正确或调整导入路径。")
    st.write("当前 sys.path:", sys.path)
    st.write("当前工作目录:", os.getcwd())
    st.stop()

# --- 配置与常量 ---
RAW_DATA_DIR = os.path.join(project_root, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(project_root, "data", "processed")
MODEL_DIR = os.path.join(project_root, "models")
REPORTS_FIGURES_DIR = os.path.join(project_root, "reports", "figures")
DEFAULT_DATA_FILE = "Online Retail.xlsx" # 默认数据集文件名

# 确保必要的目录存在
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- 应用状态管理 (使用 st.session_state) ---
def initialize_session_state():
    """初始化 Streamlit会话状态变量。"""
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'plotter' not in st.session_state:
        st.session_state.plotter = Plotter() # 初始化绘图器一次
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None # 数据加载后初始化
    if 'X_test_data' not in st.session_state:
        st.session_state.X_test_data = None
    if 'y_test_data' not in st.session_state:
        st.session_state.y_test_data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'last_trained_model_name' not in st.session_state:
        st.session_state.last_trained_model_name = None
    if 'last_saved_model_path' not in st.session_state:
        st.session_state.last_saved_model_path = None

initialize_session_state()

# --- 辅助函数 ---
def load_data_from_path(file_path):
    """从指定路径加载数据并更新会话状态。"""
    try:
        data_processor = DataProcessor(data_path=file_path)
        st.session_state.data_processor = data_processor
        st.session_state.raw_df = data_processor.load_data()
        if st.session_state.raw_df is not None:
            st.success(f"数据 '{os.path.basename(file_path)}' 加载成功!")
            st.session_state.processed_df = None # 重置已处理数据
            st.session_state.trained_model = None # 重置模型
            st.session_state.predictions = None # 重置预测结果
            return True
        else:
            st.error(f"数据处理器未能从 {file_path} 加载数据。")
            return False
    except Exception as e:
        st.error(f"加载数据 '{os.path.basename(file_path)}' 时出错: {e}")
        st.session_state.raw_df = None
        return False

def load_sample_data():
    """从本地缓存加载在线零售数据集，如果本地没有则尝试从UCI获取。"""
    sample_data_path = os.path.join(RAW_DATA_DIR, DEFAULT_DATA_FILE)
    if os.path.exists(sample_data_path):
        st.info(f"找到本地数据文件: '{sample_data_path}'。正在加载...")
        load_data_from_path(sample_data_path)
    else:
        st.info(f"'{DEFAULT_DATA_FILE}' 在 '{RAW_DATA_DIR}' 中未找到。正在尝试从 UCI 代码库获取...")
        try:
            from ucimlrepo import fetch_ucirepo
            online_retail_uci = fetch_ucirepo(id=352) # Online Retail UCI ID
            df_X = online_retail_uci.data.features

            if df_X is not None and not df_X.empty:
                try:
                    df_X.to_excel(sample_data_path, index=False) # 保存为 Excel
                    st.success(f"数据集已从 UCI 获取并保存到 {sample_data_path}")
                    load_data_from_path(sample_data_path) # 从新保存的文件加载
                except Exception as e:
                    st.warning(f"无法将获取的数据保存到 {sample_data_path}: {e}。您可能需要手动下载。")
                    st.info("将尝试使用内存中的数据...")
                    st.session_state.raw_df = df_X # 使用内存中的数据
                    # 注意: DataProcessor 通常需要一个路径。如果保存失败，某些功能可能受限。
                    # 创建一个临时的 DataProcessor 实例用于基本操作，如果需要的话。
                    # 或者提示用户一些依赖于路径的功能可能无法使用。
                    st.session_state.data_processor = None # 表示DataProcessor未从路径正确初始化
                    st.success("已加载内存中的示例数据。")

            else:
                st.error("从 UCI 代码库获取数据特征失败或数据为空。")
                st.session_state.raw_df = None
        except ImportError:
            st.error("未找到 ucimlrepo 包。请安装: pip install ucimlrepo")
        except Exception as e:
            st.error(f"从 UCI 获取数据集时出错: {e}。请尝试手动下载。")
            st.markdown(f"您可以从 [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) 下载它，并将 '{DEFAULT_DATA_FILE}' 文件（通常是 .xlsx 或 .csv）放入 '{RAW_DATA_DIR}' 目录中。")
            st.session_state.raw_df = None

# --- UI 部分 ---
st.set_page_config(layout="wide", page_title="电商智慧库存系统")
st.title("🛒 电商智慧库存系统")

# --- 侧边栏导航/控制 ---
st.sidebar.header("控制面板")
app_mode = st.sidebar.selectbox(
    "选择功能模块",
    ["数据概览与处理", "需求预测建模", "库存可视化", "关于"]
)

if st.sidebar.button("加载在线零售数据集 (示例)"):
    load_sample_data()

# --- 基于模式的主页面内容 ---
if app_mode == "数据概览与处理":
    st.header("📊 数据概览与处理")
    
    uploaded_file = st.file_uploader("上传您的销售数据 (CSV 或 XLSX)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_path = os.path.join(RAW_DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"文件 '{uploaded_file.name}' 已保存到 {file_path}")
        load_data_from_path(file_path)

    if st.session_state.get('raw_df') is not None:
        st.subheader("原始数据预览")
        st.dataframe(st.session_state.raw_df.head())
        st.write(f"原始数据形状: {st.session_state.raw_df.shape}")

        if st.session_state.data_processor:
            st.markdown("#### 数据清洗选项")
            col1, col2 = st.columns(2)
            with col1:
                remove_cancelled = st.checkbox("移除已取消/负数数量的订单?", value=True, key='cb_remove_cancelled')
                handle_outliers_qty = st.checkbox("处理'数量(Quantity)'列的异常值(IQR)?", value=False, key='cb_handle_outliers_qty')
            with col2:
                remove_zero_price = st.checkbox("移除单价(UnitPrice)为0或负数的商品?", value=True, key='cb_remove_zero_price')
                handle_outliers_price = st.checkbox("处理'单价(UnitPrice)'列的异常值(IQR)?", value=False, key='cb_handle_outliers_price')

            if st.button("执行数据清洗与转换", key="btn_process_data"):
                with st.spinner("正在清洗与转换数据..."):
                    try:
                        cleaned_df = st.session_state.data_processor.clean_data(
                            st.session_state.raw_df.copy(), # 使用原始数据的副本
                            remove_cancelled_orders=remove_cancelled,
                            remove_zero_unit_price=remove_zero_price,
                            handle_outliers_quantity=handle_outliers_qty,
                            handle_outliers_unitprice=handle_outliers_price
                        )
                        
                        if cleaned_df is not None and not cleaned_df.empty:
                            transformed_df = st.session_state.data_processor.transform_data(cleaned_df.copy())
                            st.session_state.processed_df = transformed_df
                            st.success("数据清洗与转换完成!")
                            
                            if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
                                original_filename_stem = Path(st.session_state.data_processor.data_path).stem
                                processed_file_path_parquet = os.path.join(PROCESSED_DATA_DIR, f"{original_filename_stem}_processed.parquet")
                                st.session_state.data_processor.processed_data = st.session_state.processed_df 
                                st.session_state.data_processor.save_processed_data(processed_file_path_parquet)
                                st.info(f"处理后的数据已保存到: {processed_file_path_parquet} (Parquet格式)")
                            else:
                                st.warning("数据转换后结果为空或无效，未保存。")
                                st.session_state.processed_df = None 
                        else:
                            st.warning("数据清洗后为空，无法进行后续转换。请检查清洗规则或原始数据。")
                            st.session_state.processed_df = None

                    except Exception as e:
                        st.error(f"数据处理过程中发生错误: {e}")
                        st.exception(e) 
                        st.session_state.processed_df = None 
            
            if st.session_state.get('processed_df') is not None:
                if not st.session_state.processed_df.empty:
                    st.subheader("处理后数据预览")
                    st.dataframe(st.session_state.processed_df.head())
                    st.write(f"处理后数据形状: {st.session_state.processed_df.shape}")
                else:
                    st.info("数据已处理，但结果数据集为空。")
        else:
            st.warning("数据处理器未初始化。请先加载数据。")
    else:
        st.info("请先加载或上传数据以进行处理。")

elif app_mode == "需求预测建模":
    st.header("📈 需求预测建模")
    if st.session_state.get('processed_df') is not None and not st.session_state.processed_df.empty:
        processed_df = st.session_state.processed_df
        
        st.subheader("选择建模方法和参数")
        model_type = st.selectbox("选择模型类型", ["RandomForest", "SARIMA"], key="model_type_select")
        
        # 获取数据中的日期列和数值列用于模型选择
        datetime_cols = processed_df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()

        if not datetime_cols:
            st.warning("处理后的数据中未找到日期时间列。SARIMA模型可能无法正常工作。")
        
        target_column = st.selectbox("选择目标变量 (需要预测的列，通常是销量或数量)", numeric_cols, key="target_col_model")

        # --- RandomForest 特定参数 ---
        if model_type == 'RandomForest':
            st.markdown("#### RandomForest 参数")
            # 自动排除日期列和目标列作为默认特征候选
            potential_feature_cols_rf = [col for col in numeric_cols + processed_df.select_dtypes(include=['object','category']).columns.tolist() 
                                         if col != target_column and col not in datetime_cols]
            
            # 尝试自动选择一些常见的日期衍生特征（如果存在）
            date_derived_features = ['OrderYear', 'OrderMonth', 'OrderDay', 'OrderDayOfWeek', 'OrderHour', 'WeekOfYear', 'OrderQuarter']
            default_features_rf = [col for col in date_derived_features if col in processed_df.columns and col in potential_feature_cols_rf]
            # 如果没有日期衍生特征，可以考虑其他数值或分类特征
            if not default_features_rf and potential_feature_cols_rf:
                 default_features_rf = potential_feature_cols_rf[:min(3, len(potential_feature_cols_rf))] # 最多选3个

            feature_columns_rf = st.multiselect("选择特征列 (Features)", potential_feature_cols_rf, default=default_features_rf, key="rf_features")
            
            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                n_estimators_rf = st.number_input("决策树数量 (n_estimators)", min_value=10, max_value=500, value=100, step=10, key="rf_n_est")
                max_depth_rf = st.number_input("最大深度 (max_depth)", min_value=1, max_value=50, value=10, step=1, key="rf_max_depth")
            with col_rf2:
                random_state_rf = st.number_input("随机种子 (random_state)", value=42, step=1, key="rf_rand_state")
                test_size_rf = st.slider("测试集比例 (test_size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="rf_test_size")

        # --- SARIMA 特定参数 ---
        elif model_type == "SARIMA":
            if not datetime_cols:
                st.error("SARIMA 模型需要一个日期时间类型的索引或列。请确保数据已正确处理。")
            else:
                # SARIMA 通常作用于一个以日期时间为索引的单变量序列
                # 提示用户选择日期列作为索引（如果数据中已有多个日期列）
                date_col_for_index = None
                if 'InvoiceDate' in datetime_cols: # 优先使用原始发票日期
                    date_col_for_index = 'InvoiceDate'
                elif datetime_cols: # 否则选择第一个找到的日期列
                    date_col_for_index = datetime_cols[0]

                if date_col_for_index:
                    st.info(f"SARIMA 模型将尝试使用 '{date_col_for_index}' 列作为时间索引，并对 '{target_column}' 进行预测。")
                    st.write("请确保您的目标列是按此日期聚合的时间序列数据（例如，每日/每周/每月总销售额）。")
                else:
                    st.warning("无法自动确定用于SARIMA的日期列。")

                # SARIMA 参数 (p,d,q) 和 (P,D,Q,s)
                col_sarima1, col_sarima2, col_sarima3 = st.columns(3)
                with col_sarima1:
                    p = st.number_input("阶数 p (AR)", min_value=0, value=1, key="sarima_p")
                    d = st.number_input("差分阶数 d (I)", min_value=0, value=1, key="sarima_d")
                    q = st.number_input("阶数 q (MA)", min_value=0, value=1, key="sarima_q")
                with col_sarima2:
                    P = st.number_input("季节性阶数 P", min_value=0, value=1, key="sarima_P")
                    D = st.number_input("季节性差分阶数 D", min_value=0, value=0, key="sarima_D")
                    Q = st.number_input("季节性阶数 Q", min_value=0, value=1, key="sarima_Q")
                with col_sarima3:
                    s = st.number_input("季节性周期 s (例如，月度数据为12，季度数据为4)", min_value=1, value=12, key="sarima_s")
                
                # 外生变量选择 (可选)
                potential_exog_cols_sarima = [col for col in numeric_cols if col != target_column]
                exog_cols_sarima = st.multiselect("选择外生变量 (Exogenous Variables) (可选)", potential_exog_cols_sarima, key="sarima_exog")
                
                test_size_sarima_steps = st.number_input("用于测试的期数 (例如，最后12期)", min_value=1, value=12, step=1, key="sarima_test_steps")

        if st.button(f"训练 {model_type} 模型", key=f"btn_train_{model_type}"):
            if not target_column:
                st.error("请选择一个目标变量!")
            else:
                with st.spinner(f"正在准备数据并训练 {model_type} 模型..."):
                    try:
                        model_name = f"{model_type}_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                        forecaster = None
                        y_test_actual = None
                        predictions = None
                        
                        if model_type == "RandomForest":
                            if not feature_columns_rf:
                                st.error("RandomForest: 请至少选择一个特征列。")
                            else:
                                forecaster = SalesForecaster(model_type='RandomForest', model_params={
                                    'n_estimators': n_estimators_rf, 
                                    'max_depth': max_depth_rf, 
                                    'random_state': random_state_rf
                                })
                                X_rf, y_rf = forecaster.prepare_features(processed_df, target_column=target_column, feature_columns=feature_columns_rf)
                                X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=test_size_rf, random_state=random_state_rf, shuffle=False)
                                forecaster.train(y_train_rf, X_train_rf)
                                predictions = forecaster.predict(X_test_rf)
                                y_test_actual = y_test_rf

                        elif model_type == "SARIMA":
                            if not date_col_for_index:
                                st.error("SARIMA: 无法确定日期列，无法继续。")
                            else:
                                sarima_df_indexed = processed_df.set_index(date_col_for_index) # 原始数据设置索引
                                
                                resample_freq_sarima = st.text_input("目标时间序列频率 (例如 'D', 'W', 'MS')", value='MS', key="sarima_data_freq_input")
                                aggregation_method_target = st.selectbox(f"目标列 '{target_column}' 的聚合方法", ['sum', 'mean', 'median', 'first', 'last'], index=0, key="sarima_agg_target_method")

                                if not resample_freq_sarima:
                                    st.error("请输入有效的时间序列频率。")
                                else:
                                    try:
                                        # 1. 聚合目标变量到指定频率，确保索引唯一
                                        y_sarima_resampled = sarima_df_indexed[target_column].resample(resample_freq_sarima).agg(aggregation_method_target)
                                        y_sarima = y_sarima_resampled.fillna(method='ffill') # 填充NA值
                                        
                                        st.write(f"聚合及填充后的目标序列 (y_sarima) 索引是否有重复: {y_sarima.index.has_duplicates}")
                                        st.dataframe(y_sarima.head())

                                        if y_sarima.index.has_duplicates:
                                            st.error("错误：聚合后的目标序列索引仍然存在重复值。请检查数据和聚合逻辑。")
                                            st.stop() # 停止执行，因为后续会出错

                                        # 调整后的数据点检查条件
                                        min_points_needed = max(s, d if d else 0, D * s if D and s else 0) + test_size_sarima_steps + 1 # 确保d和D*s有默认值
                                        if len(y_sarima) < min_points_needed:
                                             st.warning(f"警告: 聚合后数据点 ({len(y_sarima)}) 较少。模型可能无法稳定训练或结果不可靠。建议至少需要 {min_points_needed} 点。")
                                             # 不再直接 st.error 和 st.stop()，而是允许继续但给出警告
                                        # else:
                                        #     st.success(f"聚合后数据点 ({len(y_sarima)}) 满足最低要求 ({min_points_needed} 点)。")

                                        # 即使数据点较少，也尝试继续进行训练和测试集的划分
                                        if len(y_sarima) <= test_size_sarima_steps:
                                            st.error(f"聚合后数据点 ({len(y_sarima)}) 不足以覆盖测试期数 ({test_size_sarima_steps})。无法继续。")
                                            st.stop()
                                        else:
                                            y_train_sarima = y_sarima[:-test_size_sarima_steps]
                                            y_test_actual = y_sarima[-test_size_sarima_steps:]
                                            
                                            X_train_sarima_exog = None
                                            X_test_sarima_exog = None

                                            if exog_cols_sarima:
                                                X_sarima_exog_list = []
                                                temp_exog_df = sarima_df_indexed[exog_cols_sarima] # 从原始带索引的DataFrame取外生变量

                                                for ex_col in exog_cols_sarima:
                                                    aggregation_method_exog = st.selectbox(f"外生变量 '{ex_col}' 的聚合方法", ['mean', 'sum', 'median', 'first', 'last'], index=0, key=f"sarima_exog_agg_{ex_col}_method")
                                                    exog_col_resampled = temp_exog_df[ex_col].resample(resample_freq_sarima).agg(aggregation_method_exog)
                                                    exog_col_filled = exog_col_resampled.fillna(method='ffill')
                                                    X_sarima_exog_list.append(exog_col_filled.rename(ex_col))
                                                
                                                if X_sarima_exog_list:
                                                    X_sarima_exog_df_full = pd.concat(X_sarima_exog_list, axis=1)
                                                    st.write(f"聚合及填充后的外生变量 (X_sarima_exog_df_full) 索引是否有重复: {X_sarima_exog_df_full.index.has_duplicates}")
                                                    st.dataframe(X_sarima_exog_df_full.head())

                                                    if X_sarima_exog_df_full.index.has_duplicates:
                                                        st.error("错误：聚合后的外生变量索引存在重复值。")
                                                        st.stop()

                                                    # 对齐 y_sarima 和 X_sarima_exog_df_full 的索引
                                                    common_idx = y_sarima.index.intersection(X_sarima_exog_df_full.index)
                                                    y_sarima_aligned = y_sarima.loc[common_idx]
                                                    X_sarima_exog_df_aligned = X_sarima_exog_df_full.loc[common_idx]

                                                    # 调整后的与外生变量对齐后的数据点检查
                                                    min_points_needed_aligned = max(s, d if d else 0, D * s if D and s else 0) + test_size_sarima_steps + 1
                                                    if len(y_sarima_aligned) < min_points_needed_aligned:
                                                        st.warning(f"警告: 与外生变量对齐索引后，数据点 ({len(y_sarima_aligned)}) 较少。模型可能无法稳定训练或结果不可靠。建议至少需要 {min_points_needed_aligned} 点。")
                                                    # else:
                                                    #    st.success(f"与外生变量对齐后数据点 ({len(y_sarima_aligned)}) 满足最低要求 ({min_points_needed_aligned} 点)。")
                                                    
                                                    if len(y_sarima_aligned) <= test_size_sarima_steps:
                                                        st.error(f"与外生变量对齐索引后，数据点 ({len(y_sarima_aligned)}) 不足以覆盖测试期数 ({test_size_sarima_steps})。无法继续。")
                                                        st.stop()
                                                    else:
                                                        y_train_sarima = y_sarima_aligned[:-test_size_sarima_steps]
                                                        y_test_actual = y_sarima_aligned[-test_size_sarima_steps:]
                                                        X_train_sarima_exog = X_sarima_exog_df_aligned.loc[y_train_sarima.index]
                                                        X_test_sarima_exog = X_sarima_exog_df_aligned.loc[y_test_actual.index]
                                            
                                            # 确保 y_train_sarima 传递给 forecaster
                                            st.write("传递给模型训练的 y_train_sarima 索引是否有重复:", y_train_sarima.index.has_duplicates)
                                            if X_train_sarima_exog is not None:
                                                st.write("传递给模型训练的 X_train_sarima_exog 索引是否有重复:", X_train_sarima_exog.index.has_duplicates)

                                            forecaster = SalesForecaster(model_type='SARIMA', 
                                                                 order=(p,d,q), 
                                                                 seasonal_order=(P,D,Q,s), 
                                                                 exog_cols=exog_cols_sarima if exog_cols_sarima else None)
                                            forecaster.train(y_train_sarima, X_train_sarima_exog) # y_train_sarima 应该是聚合后的
                                            predictions = forecaster.predict(steps=len(y_test_actual), X_test=X_test_sarima_exog)
                                            if isinstance(predictions, pd.Series) and isinstance(y_test_actual, pd.Series):
                                                predictions.index = y_test_actual.index 
                                    except Exception as data_prep_ex:
                                        st.error(f"SARIMA数据准备/聚合时发生严重错误: {data_prep_ex}")
                                        st.exception(data_prep_ex)

                        if predictions is not None:
                            st.session_state.trained_model = {
                                'forecaster': forecaster,
                                'y_test': y_test_actual,
                                'predictions': predictions,
                                'model_type': model_type,
                                'target_column': target_column 
                            }
                            st.session_state.predictions = predictions
                            st.session_state.last_trained_model_name = model_name # Store model name
                            st.success(f"{model_type} 模型训练完成!")
                        else:
                            st.error(f"{model_type} 模型训练失败。")
                            st.session_state.last_trained_model_name = None # Reset if failed
                    except Exception as e:
                        st.error(f"模型训练过程中发生主要错误: {e}")
                        st.exception(e)
                        st.session_state.last_trained_model_name = None # Reset on error

        if st.session_state.get('predictions') is not None:
            st.subheader("预测结果")
            # Ensure predictions is a DataFrame or Series that st.dataframe can handle
            if isinstance(st.session_state.predictions, (pd.Series, pd.DataFrame, np.ndarray)):
                st.dataframe(st.session_state.predictions)
            if hasattr(st.session_state.predictions, 'shape'):
                st.write(f"预测结果形状: {st.session_state.predictions.shape}")
            else:
                st.write("预测结果的格式无法直接显示。")


            _forecaster_for_display_and_save = None
            if st.session_state.get('trained_model') and st.session_state.trained_model.get('forecaster'):
                _forecaster_for_display_and_save = st.session_state.trained_model['forecaster']
            
            if _forecaster_for_display_and_save:
                if _forecaster_for_display_and_save.model_type == 'RandomForest':
                    st.subheader("特征重要性")
                    feature_importances_df = _forecaster_for_display_and_save.get_feature_importances()
                    if feature_importances_df is not None and not feature_importances_df.empty:
                        # Ensure 'Feature' column exists before setting it as index
                        if 'Feature' in feature_importances_df.columns:
                             st.bar_chart(feature_importances_df.set_index('Feature'))
                        else:
                             st.warning("特征重要性数据框中缺少 'Feature' 列。")
                    else:
                        st.info("未能获取 RandomForest 模型的特征重要性。")
                elif _forecaster_for_display_and_save.model_type == 'SARIMA':
                    st.subheader("模型诊断")
                    _forecaster_for_display_and_save.plot_diagnostics(st) # Pass st object

            if st.button("保存模型"):
                if st.session_state.get('trained_model') and st.session_state.trained_model.get('forecaster') and st.session_state.get('last_trained_model_name'):
                    _forecaster_to_save = st.session_state.trained_model['forecaster']
                    _model_name_to_save = st.session_state.last_trained_model_name
                    
                    _file_extension = ".joblib" if _forecaster_to_save.model_type == "RandomForest" else ".pkl"
                    _actual_model_save_path = os.path.join(MODEL_DIR, f"{_model_name_to_save}{_file_extension}")

                    _forecaster_to_save.save_model(_actual_model_save_path)
                    st.success(f"模型已保存到: {_actual_model_save_path}")
                    st.session_state.last_saved_model_path = _actual_model_save_path
                else:
                    st.warning("没有足够信息（模型实例、forecaster 或模型名称）来保存模型。请先成功训练一个模型。")
            
            _default_path_for_load_input = st.session_state.get('last_saved_model_path', "")
            if not _default_path_for_load_input and st.session_state.get('last_trained_model_name'):
                _temp_model_info = st.session_state.get('trained_model', {})
                _temp_model_type = _temp_model_info.get('model_type', "RandomForest") # Default if no info
                _temp_ext = ".joblib" if _temp_model_type == "RandomForest" else ".pkl"
                _default_path_for_load_input = os.path.join(MODEL_DIR, f"{st.session_state.last_trained_model_name}{_temp_ext}")
            
            if not _default_path_for_load_input:
                try:
                    model_files = [
                        os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) 
                        if os.path.isfile(os.path.join(MODEL_DIR, f)) and (f.endswith(".pkl") or f.endswith(".joblib"))
                    ]
                    if model_files:
                        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        _default_path_for_load_input = model_files[0]
                except Exception:
                    pass # Silently ignore

            model_load_path_from_input = st.text_input("输入模型文件路径", value=_default_path_for_load_input, key="model_load_path_text_input_v3")

            if st.button("加载模型"):
                final_model_load_path = model_load_path_from_input
                if os.path.exists(final_model_load_path):
                    filename_lower = Path(final_model_load_path).name.lower()
                    ui_selected_model_type = st.session_state.get("model_type_select", "RandomForest")
                    
                    determined_model_type_for_load = ui_selected_model_type # Default to UI selection
                    if ".pkl" in filename_lower and "sarima" in filename_lower:
                        determined_model_type_for_load = "SARIMA"
                    elif ".joblib" in filename_lower and ("randomforest" in filename_lower or "rf" in filename_lower):
                        determined_model_type_for_load = "RandomForest"
                    elif ".pkl" in filename_lower: # Could be SARIMA or other statsmodels
                        determined_model_type_for_load = "SARIMA" # Assume SARIMA for .pkl if not clearly RF
                    elif ".joblib" in filename_lower: # Could be RF or other sklearn
                         determined_model_type_for_load = "RandomForest"


                    loaded_forecaster_instance = SalesForecaster(model_type=determined_model_type_for_load)
                    try:
                        loaded_forecaster_instance.load_model(final_model_load_path)
                        
                        st.session_state.trained_model = {
                            'forecaster': loaded_forecaster_instance,
                            'y_test': None, 
                            'predictions': None, 
                            'model_type': loaded_forecaster_instance.model_type,
                            'target_column': getattr(loaded_forecaster_instance, 'target_column', None) # if stored
                        }
                        st.session_state.predictions = None
                        st.session_state.last_trained_model_name = Path(final_model_load_path).stem
                        st.session_state.last_saved_model_path = final_model_load_path

                        st.success(f"模型 ({loaded_forecaster_instance.model_type}) 已从 {final_model_load_path} 加载。")
                        st.info("如需评估或查看预测，请使用新数据或重新运行预测步骤。")
                    except Exception as e_load:
                        st.error(f"加载模型 {final_model_load_path} 时出错: {e_load}")
                    st.exception(e_load)
                else:
                    st.error(f"文件 {final_model_load_path} 不存在。")

elif app_mode == "库存可视化":
    st.header("📊 库存可视化")
    if st.session_state.get('processed_df') is not None and not st.session_state.processed_df.empty:
        processed_df = st.session_state.processed_df
        
        st.subheader("选择可视化类型和参数")
        plot_type = st.selectbox("选择可视化类型", ["时间序列分解", "销量趋势", "库存水平", "其他"], key="plot_type_select")
        
        if plot_type == "时间序列分解":
            datetime_cols = processed_df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
            if datetime_cols:
                date_col_for_decompose = st.selectbox("选择日期列", datetime_cols, key="decompose_date_col")
                
                numeric_cols_for_decompose = processed_df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols_for_decompose:
                    st.error("处理后的数据中没有找到可用的数值列作为目标列进行时间序列分解。请检查数据处理步骤。")
                    st.stop()
                target_col_for_decompose = st.selectbox("选择数值型目标列进行分解", numeric_cols_for_decompose, key="decompose_target_col")
                
                period_for_decompose = st.number_input("输入季节性周期 (例如，月度数据填12，周度数据填7)", min_value=2, value=12, step=1, key="decompose_period")

                st.markdown("--- Optional: Resampling ---")
                apply_resampling = st.checkbox("对数据进行重采样?", value=False, key="decompose_apply_resample")
                resample_rule = None
                resample_agg_method = None

                if apply_resampling:
                    resample_rule = st.selectbox(
                        "选择重采样频率", 
                        options=['D', 'W', 'MS', 'M', 'QS', 'Q', 'AS', 'A'], 
                        index=2, 
                        key="decompose_resample_rule",
                        help="D: Daily, W: Weekly, MS: Month Start, M: Month End, QS: Quarter Start, Q: Quarter End, AS: Year Start, A: Year End"
                    )
                    resample_agg_method = st.selectbox(
                        "选择聚合方法", 
                        options=['sum', 'mean', 'median', 'first', 'last', 'count'], 
                        index=0,
                        key="decompose_resample_agg"
                    )
                
                if st.button("生成时间序列分解图"):
                    if target_col_for_decompose and date_col_for_decompose:
                        if not pd.api.types.is_numeric_dtype(processed_df[target_col_for_decompose]):
                            st.error(f"选择的目标列 '{target_col_for_decompose}' 不是数值类型。请选择一个数值列或在数据处理阶段转换该列。")
                        else:
                            with st.spinner("正在生成时间序列分解图..."):
                                try:
                                    data_to_decompose = processed_df.copy()

                                    if apply_resampling and resample_rule and resample_agg_method:
                                        st.info(f"正在将数据按 '{date_col_for_decompose}' 列以频率 '{resample_rule}' 使用 '{resample_agg_method}' 方法重采样...")
                                        try:
                                            resample_df_indexed = data_to_decompose.set_index(date_col_for_decompose)
                                            resampled_series = resample_df_indexed[target_col_for_decompose].resample(resample_rule).agg(resample_agg_method)
                                            
                                            data_to_decompose = pd.DataFrame(resampled_series)
                                            data_to_decompose = data_to_decompose.reset_index() 
                                            date_col_for_plotter = data_to_decompose.columns[0] 
                                            target_col_for_plotter = target_col_for_decompose 
                                            st.success("数据重采样完成。")
                                            st.dataframe(data_to_decompose.head())
                                        except Exception as resample_ex:
                                            st.error(f"数据重采样过程中发生错误: {resample_ex}")
                                            st.exception(resample_ex)
                                            st.stop()
                                    else:
                                        date_col_for_plotter = date_col_for_decompose
                                        target_col_for_plotter = target_col_for_decompose

                                    if 'plotter' not in st.session_state:
                                        st.session_state.plotter = Plotter()
                                    
                                    fig_decompose = st.session_state.plotter.plot_time_series_decomposition(
                                        data_to_decompose, 
                                        date_column=date_col_for_plotter, 
                                        value_column=target_col_for_plotter, 
                                        period=period_for_decompose
                                    )
                                    if fig_decompose:
                                        st.pyplot(fig_decompose)
                                    else:
                                        st.warning("未能生成时间序列分解图。请检查数据和参数，或查看控制台输出。")
                                except Exception as e:
                                    st.error(f"生成时间序列分解图时出错: {e}")
                                    st.exception(e)
            else:
                st.warning("处理后的数据中未找到日期时间列。无法生成时间序列分解图。")

        elif plot_type == "销量趋势":
            datetime_cols = processed_df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
            numeric_cols_for_trend = processed_df.select_dtypes(include=np.number).columns.tolist()

            if datetime_cols and numeric_cols_for_trend:
                date_col_for_trend = st.selectbox("选择日期列", datetime_cols, key="trend_date_col")
                
                target_col_for_trend = st.selectbox(
                    "选择要观察趋势的数值目标列", 
                    numeric_cols_for_trend, 
                    key="trend_target_col_numeric",
                    help="例如 Quantity, UnitPrice, TotalPrice"
                )
                
                agg_method_trend = st.selectbox(
                    "选择按日期聚合目标列的方法",
                    options=['sum', 'mean', 'median', 'count'], 
                    index=0, 
                    key="trend_agg_method",
                    help="例如，选择 'sum' 可以查看每日/每周/每月总销量/总销售额。"
                )

                resample_freq_trend = st.text_input(
                    "输入聚合的时间频率 (可选, 例如 'D', 'W', 'MS')", 
                    value='', 
                    key="trend_resample_freq",
                    help="留空则按原始日期中的每一天聚合。D: Daily, W: Weekly, MS: Month Start"
                ).strip()

                if st.button("生成销量趋势图"):
                    if date_col_for_trend and target_col_for_trend and agg_method_trend:
                        with st.spinner("正在生成销量趋势图..."):
                            try:
                                trend_data_df = processed_df.copy()
                                
                                if not pd.api.types.is_datetime64_any_dtype(trend_data_df[date_col_for_trend]):
                                    trend_data_df[date_col_for_trend] = pd.to_datetime(trend_data_df[date_col_for_trend], errors='coerce')
                                    trend_data_df.dropna(subset=[date_col_for_trend], inplace=True)

                                trend_data_indexed = trend_data_df.set_index(date_col_for_trend)
                                
                                if resample_freq_trend: 
                                    st.info(f"按频率 '{resample_freq_trend}' 重采样数据...")
                                    trend_aggregated_series = trend_data_indexed[target_col_for_trend].resample(resample_freq_trend).agg(agg_method_trend)
                                else: 
                                    st.info(f"按 '{date_col_for_trend}' 中的唯一日期聚合数据...")
                                    trend_aggregated_series = trend_data_indexed.groupby(pd.Grouper(level=date_col_for_trend, freq='D'))[target_col_for_trend].agg(agg_method_trend)
                                    if agg_method_trend in ['sum', 'count']:
                                         trend_aggregated_series = trend_aggregated_series[trend_aggregated_series != 0]
                                    elif agg_method_trend in ['mean', 'median']:
                                         trend_aggregated_series = trend_aggregated_series.dropna()

                                if trend_aggregated_series.empty:
                                    st.warning("聚合后的数据为空，无法生成趋势图。请检查您的选择或数据。")
                                else:
                                    trend_plot_df = pd.DataFrame(trend_aggregated_series).reset_index()
                                    date_col_for_plotter_trend = trend_plot_df.columns[0]
                                    target_col_for_plotter_trend = target_col_for_trend

                                    st.dataframe(trend_plot_df.head()) 

                                    if 'plotter' not in st.session_state:
                                        st.session_state.plotter = Plotter()
                                    
                                    fig_trend = st.session_state.plotter.plot_sales_trend(
                                        trend_plot_df, 
                                        date_column=date_col_for_plotter_trend, 
                                        sales_column=target_col_for_plotter_trend
                                    )
                                    if fig_trend:
                                        st.pyplot(fig_trend)
                                    else:
                                        st.warning("未能生成销量趋势图。请检查数据和参数。")
                            except Exception as e:
                                st.error(f"生成销量趋势图时出错: {e}")
                                st.exception(e)
                    else:
                        st.warning("请确保已选择日期列、数值目标列和聚合方法。")
            elif not datetime_cols:
                st.warning("处理后的数据中未找到日期时间列。无法生成销量趋势图。")
            elif not numeric_cols_for_trend:
                st.warning("处理后的数据中未找到数值列作为目标。无法生成销量趋势图。")

        elif plot_type == "库存水平":
            st.subheader("当前库存水平估算与可视化")
            if st.session_state.get('processed_df') is not None and not st.session_state.processed_df.empty:
                processed_df_stock = st.session_state.processed_df

                if not st.session_state.get('data_processor'):
                    st.warning('DataProcessor 未初始化，某些功能可能受限。请确保数据已通过"数据概览与处理"正确加载和处理。')
                    st.stop()
                
                data_processor_instance = st.session_state.data_processor

                st.markdown("#### 1. 选择数据列")
                product_id_cols_stock = [col for col in processed_df_stock.columns if processed_df_stock[col].dtype == 'object' or pd.api.types.is_string_dtype(processed_df_stock[col])]
                default_prod_id_stock = None
                if 'StockCode' in product_id_cols_stock: default_prod_id_stock = 'StockCode'
                elif 'Description' in product_id_cols_stock: default_prod_id_stock = 'Description'
                
                product_id_col_stock = st.selectbox(
                    "选择产品标识列", 
                    options=product_id_cols_stock, 
                    index=product_id_cols_stock.index(default_prod_id_stock) if default_prod_id_stock and default_prod_id_stock in product_id_cols_stock else 0,
                    key="stock_product_id_col",
                    help="通常是 StockCode 或 Description"
                )
                
                qty_cols_stock = processed_df_stock.select_dtypes(include=np.number).columns.tolist()
                default_qty_col_stock = None
                if 'Quantity' in qty_cols_stock: default_qty_col_stock = 'Quantity'

                quantity_col_stock = st.selectbox(
                    "选择交易数量列", 
                    options=qty_cols_stock, 
                    index=qty_cols_stock.index(default_qty_col_stock) if default_qty_col_stock and default_qty_col_stock in qty_cols_stock else 0,
                    key="stock_quantity_col",
                    help="用于计算已售数量"
                )

                st.markdown("#### 2. 设定初始库存 (估算用)")
                assumed_initial_stock = st.number_input(
                    "为所有产品设定一个统一的假定初始库存值", 
                    min_value=0, 
                    value=1000,
                    step=50, 
                    key="stock_initial_assumed",
                    help="这将用于从总销量中减去，以估算当前库存。这是一个简化模型。"
                )

                if st.button("计算并显示当前估算库存", key="btn_calc_stock"):
                    if product_id_col_stock and quantity_col_stock:
                        with st.spinner("正在计算产品总销量并估算库存..."):
                            total_sold_df = data_processor_instance.calculate_total_quantity_sold(
                                processed_df_stock, 
                                product_id_col=product_id_col_stock, 
                                quantity_col=quantity_col_stock
                            )

                            if total_sold_df is not None:
                                if total_sold_df.empty:
                                    if not processed_df_stock[(processed_df_stock[quantity_col_stock] > 0)].empty:
                                        st.info(f"已尝试计算总销量，但未能为任何有效的产品ID ('{product_id_col_stock}') 找到正数销量记录，或者所有产品ID都为空/无效。")
                                    else:
                                        st.info(f"在数据中没有找到 '{quantity_col_stock}' 大于0的记录，因此无法计算任何销量。")
                                
                                all_products = pd.DataFrame({product_id_col_stock: processed_df_stock[product_id_col_stock].unique()})
                                all_products = all_products[all_products[product_id_col_stock].notna() & (all_products[product_id_col_stock] != '')]

                                if not total_sold_df.empty:
                                    current_stock_df = pd.merge(all_products, total_sold_df, on=product_id_col_stock, how='left')
                                else: 
                                    current_stock_df = all_products.copy()
                                    current_stock_df['TotalSold'] = 0 

                                current_stock_df['TotalSold'] = current_stock_df['TotalSold'].fillna(0).astype(int)
                                current_stock_df['EstimatedCurrentStock'] = assumed_initial_stock - current_stock_df['TotalSold']
                                
                                st.session_state.current_stock_df = current_stock_df.sort_values(by='EstimatedCurrentStock', ascending=True)
                                if not current_stock_df.empty :
                                     st.success("当前估算库存计算完成!")
                                elif total_sold_df.empty: 
                                     st.warning("没有有效的产品ID可用于库存估算。")
                                else: 
                                     st.warning("估算库存为空，请检查原始数据中的产品ID。")
                            else: 
                                st.error("计算总销量时发生错误。无法估算库存。")
                                st.session_state.current_stock_df = None
                    else:
                        st.warning("请选择产品标识列和数量列。")
                
                if 'current_stock_df' in st.session_state and st.session_state.current_stock_df is not None and not st.session_state.current_stock_df.empty:
                    st.markdown("#### 3. 库存可视化与筛选")
                    current_stock_display_df = st.session_state.current_stock_df
                    
                    view_option_stock = st.selectbox(
                        "选择查看方式",
                        options=["所有产品", "搜索特定产品", "库存最高的产品", "库存最低的产品 (可能缺货)"],
                        key="stock_view_option"
                    )

                    num_products_to_show = 10 

                    if view_option_stock == "所有产品":
                        st.dataframe(current_stock_display_df)
                    
                    elif view_option_stock == "搜索特定产品":
                        search_term_stock = st.text_input(f"输入要搜索的 {product_id_col_stock}", key="stock_search_term")
                        if search_term_stock:
                            results_df = current_stock_display_df[current_stock_display_df[product_id_col_stock].astype(str).str.contains(search_term_stock, case=False, na=False)]
                            st.dataframe(results_df)
                        else:
                            st.info(f"请输入 '{product_id_col_stock}' 进行搜索。")

                    elif view_option_stock == "库存最高的产品":
                        num_products_to_show = st.slider("显示库存最高的产品数量", 5, 50, 10, key="stock_top_n")
                        st.dataframe(current_stock_display_df.sort_values(by='EstimatedCurrentStock', ascending=False).head(num_products_to_show))
                        
                        if st.session_state.get('plotter'):
                            st.subheader(f"库存最高的 {num_products_to_show} 种产品")
                            fig_stock_top = st.session_state.plotter.plot_bar_chart(
                                current_stock_display_df.sort_values(by='EstimatedCurrentStock', ascending=False).head(num_products_to_show),
                                x_column=product_id_col_stock,
                                y_column='EstimatedCurrentStock',
                                title=f"库存最高的 {num_products_to_show} 种产品",
                                horizontal=True 
                            )
                            if fig_stock_top: st.pyplot(fig_stock_top)

                    elif view_option_stock == "库存最低的产品 (可能缺货)":
                        num_products_to_show = st.slider("显示库存最低的产品数量", 5, 50, 10, key="stock_bottom_n")
                        lowest_stock_df = current_stock_display_df.sort_values(by='EstimatedCurrentStock', ascending=True)
                        st.dataframe(lowest_stock_df.head(num_products_to_show))

                        if st.session_state.get('plotter'):
                            st.subheader(f"库存最低的 {num_products_to_show} 种产品")
                            fig_stock_bottom = st.session_state.plotter.plot_bar_chart(
                                lowest_stock_df.head(num_products_to_show).sort_values(by='EstimatedCurrentStock', ascending=False), 
                                x_column=product_id_col_stock,
                                y_column='EstimatedCurrentStock',
                                title=f"库存最低的 {num_products_to_show} 种产品 (可能缺货)",
                                horizontal=True
                            )
                            if fig_stock_bottom: st.pyplot(fig_stock_bottom)
                elif 'current_stock_df' in st.session_state and st.session_state.current_stock_df is not None and st.session_state.current_stock_df.empty:
                     st.info("估算库存结果为空。这可能发生在所有产品ID都无效，或者初始计算步骤未能生成任何数据。")
            else: 
                st.info('请先在"数据概览与处理"模块加载并处理数据，然后才能进行库存可视化。')
        elif plot_type == "其他":
            st.write("其他可视化选项待实现。")
    else:
        st.info('请先在"数据概览与处理"模块加载并成功处理数据，然后才能访问此模块。')


elif app_mode == "关于":
    st.header("📚 关于")
    st.write("这是一个电商智慧库存系统的示例应用。")
    st.write("它包括以下功能模块:")
    st.markdown("- 数据概览与处理")
    st.markdown("- 需求预测建模")
    st.markdown("- 库存可视化")
    st.write("您可以在侧边栏选择不同的功能模块。")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # 添加了更多评估指标
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

class SalesForecaster:
    """
    一个使用机器学习进行销售预测的类。
    """

    def __init__(self, model_type='RandomForest', model_params=None, 
                 order=None, seasonal_order=None, exog_cols=None): # 添加了 SARIMA 参数和 exog_cols
        """
        初始化 SalesForecaster。

        参数:
            model_type (str): 要使用的模型类型 (例如, 'RandomForest')。
            model_params (dict, optional): 模型的参数。默认为 None。
            order (tuple, optional): ARIMA/SARIMA 的阶数 (p,d,q)。默认为 None。
            seasonal_order (tuple, optional): SARIMA 的季节性阶数 (P,D,Q,s)。默认为 None。
            exog_cols (list, optional): 外生变量列名列表。默认为 None。
        """
        self.model_type = model_type
        self.model_params = model_params if model_params is not None else {}
        self.order = order # ARIMA/SARIMA 的阶数 (p,d,q)
        self.seasonal_order = seasonal_order # SARIMA 的季节性阶数 (P,D,Q,s)
        self.exog_cols = exog_cols # 外生变量列名列表
        self.model = None # 初始化模型为 None，将由 _initialize_model 设置或在 SARIMA 的 train 中设置
        self.is_trained = False
        self.training_columns = None # 用于存储 sklearn 模型的列顺序

        # 尝试初始化模型，SARIMA将在train中完全初始化
        if self.model_type != 'SARIMA':
            self.model = self._initialize_model()

    def _initialize_model(self):
        """
        初始化指定的机器学习模型。
        """
        if self.model_type == 'RandomForest':
            # 如果未提供，则应用默认参数，可以扩展
            if not self.model_params:
                self.model_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
            return RandomForestRegressor(**self.model_params)
        # SARIMA 模型在 train 方法中初始化，因为它需要数据。
        elif self.model_type == 'SARIMA':
            if self.order is None:
                self.order = (1, 1, 1) # 默认阶数
                print(f"警告: 未提供 SARIMA 阶数。使用默认值 {self.order}")
            if self.seasonal_order is None:
                self.seasonal_order = (0, 0, 0, 0) # 默认季节性阶数 (无季节性)
                print(f"警告: 未提供 SARIMA seasonal_order。使用默认值 {self.seasonal_order}")
            return None # 实际的 SARIMAX 对象在 train() 中创建
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def prepare_features(self, df: pd.DataFrame, target_column: str, feature_columns: list = None) -> tuple:
        """
        为模型训练准备特征 (X) 和目标 (y)。

        参数:
            df (pd.DataFrame): 包含特征和目标的输入 DataFrame。
            target_column (str): 目标变量列的名称。
            feature_columns (list, optional): 特征列名称列表。
            如果为 None，则使用除目标之外的所有列。

        返回:
            tuple: (X, y)，其中 X 是特征的 DataFrame，y 是目标的 Series。
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        y = df[target_column]
        X = None

        if self.model_type == 'SARIMA':
            # 对于 SARIMA，如果指定，X 将是外生变量
            if self.exog_cols:
                if all(col in df.columns for col in self.exog_cols):
                    X = df[self.exog_cols]
                else:
                    missing_exog = [col for col in self.exog_cols if col not in df.columns]
                    raise ValueError(f"Missing exogenous columns for SARIMA: {missing_exog}")
            # y 仍然是时间序列本身，通常是单个列 (target_column)
            return X, y # X can be None
        else: # 对于 RandomForest 和其他类似 sklearn 的模型
            if feature_columns:
                if not all(col in df.columns for col in feature_columns):
                    missing_feat = [col for col in feature_columns if col not in df.columns]
                    raise ValueError(f"Missing feature columns: {missing_feat}")
                X = df[feature_columns]
            else:
                X = df.drop(columns=[target_column])
            
            self.training_columns = X.columns.tolist() # 保存列顺序和名称
            return X, y
    
    

    def train(self, y_train: pd.Series, X_train: pd.DataFrame = None):
        """
        训练预测模型。

        参数:
            y_train (pd.Series): 训练目标变量。
            X_train (pd.DataFrame, optional): 训练特征。
        """
        print(f"正在训练 {self.model_type} 模型...")
        if self.model_type == 'SARIMA':
            if X_train is not None and self.exog_cols is None:
                print("警告: 为 SARIMA 提供了 X_train，但在初始化时未指定 exog_cols。除非设置了 exog_cols，否则将忽略 X_train。")
                current_exog = None
            elif X_train is not None and self.exog_cols is not None:
                if not all(col in X_train.columns for col in self.exog_cols):
                    raise ValueError("SARIMA 的 X_train 不包含所有指定的 exog_cols。")
                current_exog = X_train[self.exog_cols]
            else:
                current_exog = None # 没有外生变量
            
            # 如果可能，确保y_train是具有 DatetimeIndex 的 Series，以用于 SARIMA
            if not isinstance(y_train.index, pd.DatetimeIndex):
                warnings.warn("SARIMA y_train 没有 DatetimeIndex。模型可能无法最佳执行，或者基于时间的特征可能会被错误解释。")

            self.model = SARIMAX(endog=y_train, 
                                 exog=current_exog, 
                                 order=self.order, 
                                 seasonal_order=self.seasonal_order,
                                 enforce_stationarity=False, 
                                 enforce_invertibility=False,
                                 simple_differencing=False) # simple_differencing 用于stasmodels的pre-python 3.7.1兼容性
            
            with warnings.catch_warnings(): # 在拟合期间抑制SARIMA的警告
                warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
                warnings.filterwarnings("ignore", category=RuntimeWarning, module='statsmodels')
                try:
                    self.model = self.model.fit(disp=False) # disp=False 关闭详细输出
                except Exception as e:
                    print(f"SARIMA 模型拟合错误: {e}")
                    # 如果错误与LU分解失败或矩阵不是正定有关，则尝试使用simple_differencing=True作为回退
                    if 'LU decomposition failed' in str(e) or 'matrix is not positive definite' in str(e):
                        print("重试 SARIMA 拟合 with simple_differencing=True")
                        self.model = SARIMAX(endog=y_train, exog=current_exog, order=self.order, seasonal_order=self.seasonal_order, simple_differencing=True)
                        self.model = self.model.fit(disp=False)
                    else:
                        raise # 重新引发其他错误
        else: # 对于 RandomForest 和其他 sklearn 模型
            if X_train is None:
                raise ValueError("X_train 不能为 None 用于非 SARIMA 模型。")
            if self.training_columns is None: # 应该在 prepare_features 中设置
                self.training_columns = X_train.columns.tolist()
            self.model.fit(X_train[self.training_columns], y_train) # 确保列顺序一致
        
        self.is_trained = True
        print("模型训练完成。")

    def predict(self, X_test: pd.DataFrame = None, steps: int = None): # steps for SARIMA forecasting
        """
        使用训练好的模型进行预测。

        Args:
            X_test (pd.DataFrame, optional): 用于预测的特征。
            steps (int, optional): 用于 SARIMA 预测的步数。

        Returns:
            np.ndarray: 预测值。
        
        Raises:
            RuntimeError: 如果模型未训练。
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练。请先调用 train()。")
        print("正在进行预测...")

        if self.model_type == 'SARIMA':
            # 对于 SARIMA，X_test 包含未来外生变量（如果使用）
            current_exog_test = None
            if self.exog_cols and X_test is not None:
                if not all(col in X_test.columns for col in self.exog_cols):
                    raise ValueError("SARIMA 预测的 X_test 不包含所有指定的 exog_cols。")
                current_exog_test = X_test[self.exog_cols]
            elif self.exog_cols and X_test is None and steps is not None:
                 # 如果训练中使用了外生变量但未提供未来外生变量（X_test），这将是一个问题
                 if steps > 0: # 检查 steps 以确认这是一个预测
                    print("警告: SARIMA 模型使用外生变量训练，但未提供未来外生变量（X_test）进行多步预测。预测可能不可靠或失败。")
            
            if steps is not None:
                # 预测未来值
                predictions = self.model.forecast(steps=steps, exog=current_exog_test)
            elif X_test is not None and hasattr(self.model, 'predict'): # 样本内或特定点预测
                # 这种形式的 predict 对于 SARIMAX 通常需要 start/end 索引或 dynamic=False 用于样本内
                # 对于带有外生变量的样本外，这可能很棘手。forecast() 通常更安全用于未来。
                # 我们假设 X_test 提供了 exog 用于由其索引定义的特定点。
                if current_exog_test is not None:
                    predictions = self.model.predict(start=X_test.index[0], end=X_test.index[-1], exog=current_exog_test)
                else: # 没有外生变量，根据时间索引预测如果 X_test 索引与模型对齐
                    predictions = self.model.predict(start=X_test.index[0], end=X_test.index[-1])
            else:
                raise ValueError("对于 SARIMA，要么提供 'steps' 进行预测，要么提供 'X_test' 与索引进行特定点预测。")
            return predictions # 返回一个 pandas Series 用于 SARIMA
        else: # 对于 RandomForest 等
            if X_test is None:
                raise ValueError("X_test 不能为 None 用于非 SARIMA 模型的预测。")
            # 确保 X_test 具有相同的列和顺序作为 training_columns
            if self.training_columns is not None:
                 X_test_aligned = X_test.reindex(columns=self.training_columns, fill_value=0) # fill_value 安全如果新列出现
                 if X_test_aligned.shape[1] != len(self.training_columns):
                     raise ValueError(f"X_test columns {X_test.columns.tolist()} do not match training columns {self.training_columns}") 
                 predictions = self.model.predict(X_test_aligned)
            else:
                 predictions = self.model.predict(X_test) # 应该不会发生如果训练正确
            return predictions # 返回一个 numpy 数组用于 sklearn 模型

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict: # y_pred can be Series or Array
        """
            评估模型性能。

        Args:
            y_true (pd.Series): 真实目标值。
            y_pred (pd.Series): 预测值。

        Returns:
            dict: 包含评估指标（如 MSE、RMSE、MAE、R2、MAPE）的字典。
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE 计算
        y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
        # 避免 MAPE 除以零，用一个很小的数替换 y_true 中的 0，或者删除这对
        # 为简单起见，我们用一个非常小的数替换 0 以避免零除法，如果有很多 0，这可能会影响 MAPE
        y_true_mape = np.where(y_true_np == 0, 1e-8, y_true_np)
        mape = np.mean(np.abs((y_true_np - y_pred_np) / y_true_mape)) * 100
        
        print(f"Evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    def plot_diagnostics(self, streamlit_app_object, figsize=(15, 12)) -> None:
        """
        如果模型是 SARIMA 并且已经训练过，则显示其诊断图。
        直接在 Streamlit 应用程序对象上绘图。

        Args:
            streamlit_app_object: Streamlit 的 `st` 对象，用于绘图。
            figsize (tuple): 诊断图的尺寸。
        """
        if self.model_type == 'SARIMA' and self.is_trained and hasattr(self.model, 'plot_diagnostics'):
            print("正在为 SARIMA 模型生成诊断图...")
            try:
                # SARIMAXResultsWrapper.plot_diagnostics 返回一个 Figure 对象
                fig = self.model.plot_diagnostics(figsize=figsize)
                streamlit_app_object.pyplot(fig)
                print("SARIMA 诊断图已显示。")
            except ValueError as ve: # 更具体地捕获 ValueError
                error_message = f"绘制 SARIMA 诊断图时出错: {ve}。"
                error_message += " 这通常意味着训练数据相对于模型的阶数（p, d, q, P, D, Q, s）和季节性周期来说太短了。"
                error_message += " 请尝试减少模型阶数（例如，减小p, q, P, Q的值）、减少季节性周期s（如果适用）、关闭或减少差分（d, D），或使用更多的数据点进行训练。"
                print(error_message)
                streamlit_app_object.error(error_message)
            except Exception as e:
                error_message = f"绘制 SARIMA 诊断图时发生未知错误: {e}"
                print(error_message)
                streamlit_app_object.error(error_message)
        elif self.model_type != 'SARIMA':
            # streamlit_app_object.info("诊断图仅适用于 SARIMA 模型。")
            pass # 对于非SARIMA模型，静默处理或提供信息
        elif not self.is_trained:
            streamlit_app_object.warning("模型尚未训练，无法显示诊断图。")
        else:
            streamlit_app_object.warning("此模型对象没有可用的 plot_diagnostics 方法。")

    def get_feature_importances(self):
        """
        检索特征重要性如果模型是 RandomForest 并且已经训练过。

        Returns:
            pd.DataFrame: DataFrame 包含 'Feature' 和 'Importance'，按重要性排序。
                          如果不适用于或模型未训练，则返回 None。
        """
        if self.model_type == 'RandomForest' and self.is_trained and hasattr(self.model, 'feature_importances_'):
            if self.training_columns is None or len(self.training_columns) != len(self.model.feature_importances_):
                print("警告: 训练列信息与特征重要性不一致。")
                return None
            
            importances_df = pd.DataFrame({
                'Feature': self.training_columns,
                'Importance': self.model.feature_importances_
            })
            importances_df = importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            return importances_df
        else:
            # print("Feature importances are not available for this model type or model is not trained.") 
            return None

    def save_model(self, filepath: str):
        """
        将训练好的模型保存到文件。

        Args:
            filepath (str): 保存模型的路径。
        """
        if not self.is_trained:
            print("模型未训练。什么也不保存。")
            return
        try:
            if self.model_type == 'SARIMA':
                self.model.save(filepath) # SARIMAX has its own save method
            else:
                joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath: str, model_type_override: str = None):
        """
        从文件加载训练好的模型。

        Args:
            filepath (str): 加载模型的路径。
            model_type_override (str, optional): 如果模型类型未存储在模型中，则覆盖模型类型。
        """
        # model_type_override 在模型类型未存储在模型中时至关重要
        current_model_type = model_type_override if model_type_override else self.model_type
        if not current_model_type: # 如果 self.model_type 为 None 且没有覆盖
            raise ValueError("model_type 必须已知才能加载模型。提供 model_type_override 或设置在初始化时。")
        try:
            if current_model_type == 'SARIMA':
                from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
                self.model = SARIMAXResultsWrapper.load(filepath)
            else:
                self.model = joblib.load(filepath)
            
            self.is_trained = True
            self.model_type = current_model_type # 确保在加载后正确设置 model_type
            print(f"模型 ({self.model_type}) 从 {filepath} 加载")
            # 对于 sklearn 模型，尝试加载训练列如果单独保存或重新推断
            if hasattr(self.model, 'feature_names_in_') and self.model_type != 'SARIMA':
                 self.training_columns = self.model.feature_names_in_.tolist()
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    # 示例用法（用于测试或在此文件中演示）
if __name__ == '__main__':
    # 创建用于预测的虚拟数据
    # 这是一个非常简化的示例。实际销售数据会有时间序列特征。
    data_size = 200
    dummy_features = pd.DataFrame({
        'feature1': np.random.rand(data_size) * 100,
        'feature2': np.random.rand(data_size) * 50,
        'month': np.random.randint(1, 13, size=data_size),
        # 'product_category': np.random.choice(['Electronics', 'Clothing', 'Home'], size=data_size) # 分类特征
    })
    # 模拟目标变量（例如，销售数量）
    dummy_target = (2 * dummy_features['feature1'] - 
                    0.5 * dummy_features['feature2'] + 
                    np.random.normal(0, 10, size=data_size) + # noise
                    dummy_features['month'] * 5 # seasonal component
                   )
    dummy_target = np.maximum(0, dummy_target) # Sales cannot be negative

    # 合并成一个 DataFrame
    dummy_sales_df = dummy_features.copy()
    dummy_sales_df['sales_quantity'] = dummy_target.round().astype(int)

    print("创建了虚拟销售数据:")
    print(dummy_sales_df.head())

    # 初始化预测器
    forecaster = SalesForecaster(model_type='RandomForest')

    # 准备特征和目标
    # 在实际场景中，feature_columns 会仔细选择/工程化。
    # 如果包含 'product_category'，需要编码（例如，独热编码）。
    X, y = forecaster.prepare_features(dummy_sales_df, target_column='sales_quantity', feature_columns=['feature1', 'feature2', 'month'])

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    forecaster.train(y_train=y_train, X_train=X_train)

    # 进行预测
    predictions = forecaster.predict(X_test=X_test)
    # print("\nSample Predictions:", predictions[:5])

    # 评估模型
    evaluation_metrics = forecaster.evaluate(y_test, predictions)

    # 保存和加载模型（示例）
    model_path = '../../models/sales_forecaster_rf.joblib' # 调整路径
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    forecaster.save_model(model_path)

    loaded_forecaster = SalesForecaster(model_type='RandomForest') # 用相同的类型初始化
    loaded_forecaster.load_model(model_path)
    loaded_predictions = loaded_forecaster.predict(X_test)

    # 验证加载的模型做出相同的预测
    if np.array_equal(predictions, loaded_predictions):
        print("\n模型保存和加载成功。预测匹配。")
    else:
        print("\nError: 加载的模型做出的预测与原始预测不匹配。")

    # --- SARIMA 示例 ---
    print("\n--- SARIMA 示例 ---")
    date_rng = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    data_size_sarima = len(date_rng)
    # 模拟具有季节性和趋势的时间序列数据
    y_sarima_values = (np.arange(data_size_sarima) * 0.5 + 
                       np.sin(np.arange(data_size_sarima) * (2 * np.pi / 12)) * 10 + 
                       np.random.normal(0, 5, size=data_size_sarima) + 50)
    y_sarima_values = np.maximum(0, y_sarima_values).round()
    y_sarima = pd.Series(y_sarima_values, index=date_rng, name='monthly_sales')
    
    # 模拟外生变量
    exog_data = pd.DataFrame({
        'promo_effect': np.random.randint(0, 2, size=data_size_sarima) * 5 
    }, index=date_rng)

    # 分割数据用于 SARIMA（通常在所有过去上训练，预测未来）
    # 对于这个示例，让我们保留最后 12 个月用于测试预测与实际值
    train_end_date = y_sarima.index[-13]
    y_train_sarima = y_sarima.loc[:train_end_date]
    exog_train_sarima = exog_data.loc[:train_end_date]
    
    y_test_sarima = y_sarima.loc[y_sarima.index > train_end_date]
    exog_test_sarima = exog_data.loc[exog_data.index > train_end_date]

    # SARIMA 特定参数
    sarima_order = (1, 1, 1) # (p,d,q)
    sarima_seasonal_order = (1, 1, 0, 12) # (P,D,Q,s)

    forecaster_sarima = SalesForecaster(model_type='SARIMA', 
                                      order=sarima_order, 
                                      seasonal_order=sarima_seasonal_order,
                                      exog_cols=['promo_effect'] if exog_data is not None else None)
    
    # 对于 SARIMA，`prepare_features` 主要分离 y 和 exog 如果定义了 exog_cols
    # 实际的 y（时间序列）和 exog 直接传递给 train。
    # 这里传递组合的 df 只是为了遵循现有的 prepare_features 签名。
    # 在实际应用中，SARIMA 的数据流可能更直接。
    # temp_train_df_sarima = pd.concat([y_train_sarima, exog_train_sarima], axis=1)
    # _, _ = forecaster_sarima.prepare_features(temp_train_df_sarima, target_column='monthly_sales') 
    # 这个 prepare_features 调用对于 SARIMA 主要是为了保持一致性，如果有一个统一的流程；
    # 它将简单地返回 exog_train_sarima（如果设置了 exog_cols）和 y_train_sarima。

    print(f"Training SARIMA with y_train of length {len(y_train_sarima)} and exog_train of shape {exog_train_sarima.shape if exog_train_sarima is not None else 'None'}")
    forecaster_sarima.train(y_train=y_train_sarima, X_train=exog_train_sarima) 

    # 预测测试集的长度
    n_forecast_steps = len(y_test_sarima)
    predictions_sarima = forecaster_sarima.predict(steps=n_forecast_steps, X_test=exog_test_sarima)
    
    if predictions_sarima is not None:
        print(f"SARIMA Predictions (length {len(predictions_sarima)}):\n{predictions_sarima.head()}")
        print(f"Actual test values (length {len(y_test_sarima)}):\n{y_test_sarima.head()}")
        # 如果需要，对齐索引以进行直接比较或绘图
        if isinstance(predictions_sarima, pd.Series) and isinstance(y_test_sarima, pd.Series):
            predictions_sarima.index = y_test_sarima.index # 关键用于评估
            forecaster_sarima.evaluate(y_test_sarima, predictions_sarima)
        else:
            print("无法对齐 SARIMA 预测与测试集以进行完整评估。")
    else:
        print("SARIMA 预测失败。")

    # 保存和加载 SARIMA 模型
    sarima_model_path = '../../models/sales_forecaster_sarima.pkl' # Statsmodels uses .pkl
    forecaster_sarima.save_model(sarima_model_path)
    
    loaded_forecaster_sarima = SalesForecaster(model_type='SARIMA', order=sarima_order, seasonal_order=sarima_seasonal_order, exog_cols=['promo_effect'] if exog_data is not None else None)
    # 对于 statsmodels，加载需要模型类本身，而不是 joblib 的文件路径。
    # 现在 `load_model` 方法处理这一点。
    loaded_forecaster_sarima.load_model(sarima_model_path) # model_type_override not strictly needed if already set
    
    loaded_predictions_sarima = loaded_forecaster_sarima.predict(steps=n_forecast_steps, X_test=exog_test_sarima)
    if loaded_predictions_sarima is not None and isinstance(predictions_sarima, pd.Series):
        if predictions_sarima.equals(loaded_predictions_sarima):
             print("\nSARIMA 模型保存和加载成功。预测匹配。")
        else:
             print("\nError: 加载的 SARIMA 模型做出的预测与原始预测不完全匹配。（这有时可能会由于浮点精度问题或如果模型状态没有被所有版本的 statsmodels 完全序列化。）")
             # print("Original:", predictions_sarima.head())
             # print("Loaded:", loaded_predictions_sarima.head())
             # print("Difference:", (predictions_sarima - loaded_predictions_sarima).abs().sum())

    print("SARIMA 示例完成。") 
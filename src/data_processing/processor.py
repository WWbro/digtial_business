import pandas as pd
import numpy as np
from typing import Union, Optional

class DataProcessor:
    """
    一个用于处理电子商务数据的类，专门为在线零售数据集量身定制。
    处理诸如清理、转换和特征工程等任务。
    """

    def __init__(self, data_path: str):
        """
        使用原始数据的路径初始化 DataProcessor。

        参数:
            data_path (str): 原始数据文件的路径（例如，CSV 或 XLSX）。
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        从指定路径加载数据。
        根据扩展名支持 CSV 和 Excel 文件。

        返回:
            pd.DataFrame: 加载的 DataFrame，如果加载失败则返回 None。
        
        引发:
            FileNotFoundError: 如果 data_path 处的文件不存在。
            ValueError: 如果文件扩展名不受支持。
            Exception: 其他加载错误。
        """
        try:
            file_extension = self.data_path.split('.')[-1].lower()
            if file_extension == 'csv':
                self.raw_data = pd.read_csv(self.data_path, low_memory=False)
            elif file_extension in ['xls', 'xlsx']:
                self.raw_data = pd.read_excel(self.data_path, sheet_name=0)
            else:
                raise ValueError(f"不支持的文件扩展名: {file_extension}。请使用 CSV 或 Excel。")
            
            print(f"数据成功从 {self.data_path} 加载。形状: {self.raw_data.shape}")
            return self.raw_data
        except FileNotFoundError:
            print(f"错误: 在 {self.data_path} 未找到文件")
            raise
        except ValueError as ve:
            print(f"数据加载期间发生 ValueError: {ve}")
            raise
        except Exception as e:
            print(f"从 {self.data_path} 加载数据时出错: {e}")
            raise

    def _handle_column_outliers(self, df: pd.DataFrame, column_name: str, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """使用 IQR 方法处理特定列中的异常值的辅助函数。"""
        if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
            print(f"警告: 列 '{column_name}' 未找到或非数字类型。跳过异常值处理。")
            return df

        print(f"正在使用 {method.upper()} 方法处理列: {column_name} 的异常值...")
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        original_values_below = df[df[column_name] < lower_bound].shape[0]
        original_values_above = df[df[column_name] > upper_bound].shape[0]

        # 限制异常值 (Winsorization)
        df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
        df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])
        
        capped_below = df[df[column_name] == lower_bound].shape[0] - (original_values_below if lower_bound == Q1 - multiplier * IQR else 0)
        capped_above = df[df[column_name] == upper_bound].shape[0] - (original_values_above if upper_bound == Q3 + multiplier * IQR else 0)
        
        if original_values_below > 0:
            print(f"已将 {column_name} 中 {original_values_below} 个低于下限 ({lower_bound:.2f}) 的值限制。")
        if original_values_above > 0:
            print(f"已将 {column_name} 中 {original_values_above} 个高于上限 ({upper_bound:.2f}) 的值限制。")
        if original_values_below == 0 and original_values_above == 0:
            print(f"在 {column_name} 中未检测到需要限制的异常值，边界 L:{lower_bound:.2f}, U:{upper_bound:.2f}。")
        return df

    """
           执行针对在线零售数据集的数据清理操作。
           参数:
               df (pd.DataFrame): 要清理的 DataFrame。
               remove_cancelled_orders (bool): 如果为 True，则移除 Quantity 为负数的行（假定为已取消订单）。
               remove_zero_unit_price (bool): 如果为 True，则移除 UnitPrice <= 0 的行。
               handle_outliers_quantity (bool): 如果为 True，则使用 IQR 处理 'Quantity' 中的异常值。
               handle_outliers_unitprice (bool): 如果为 True，则使用 IQR 处理 'UnitPrice' 中的异常值。
           返回:
               pd.DataFrame: 清理后的 DataFrame。
           """
    def clean_data(self, df: pd.DataFrame, 
                   remove_cancelled_orders: bool = True, 
                   remove_zero_unit_price: bool = True,
                   handle_outliers_quantity: bool = False,
                   handle_outliers_unitprice: bool = False) -> pd.DataFrame:

        print("开始数据清理...")
        cleaned_df = df.copy()

        cleaned_df.columns = cleaned_df.columns.astype(str).str.strip()

        if 'InvoiceDate' in cleaned_df.columns:
            cleaned_df['InvoiceDate'] = pd.to_datetime(cleaned_df['InvoiceDate'], errors='coerce')
            print(f"已将 'InvoiceDate' 转换为 datetime。转换后的空值数量: {cleaned_df['InvoiceDate'].isnull().sum()}")
            cleaned_df.dropna(subset=['InvoiceDate'], inplace=True)
            print(f"丢弃无效 InvoiceDate 后的行数: {len(cleaned_df)}")
        else:
            print("警告: 未找到 'InvoiceDate' 列。")

        if 'CustomerID' in cleaned_df.columns:
            initial_customer_nulls = cleaned_df['CustomerID'].isnull().sum()
            # 在填充 NA 之前将 CustomerID 转换为字符串，因为它可能被读取为浮点数
            cleaned_df['CustomerID'] = cleaned_df['CustomerID'].astype(str)
            # 确保其为字符串类型后，用 'UNKNOWN' 填充 NA
            if initial_customer_nulls > 0: # 如果已经转换为字符串，检查 'nan' 字符串
                 cleaned_df['CustomerID'].replace('nan', 'UNKNOWN', inplace=True) # 处理 NaN 变成 'nan' 字符串的情况
            cleaned_df['CustomerID'].fillna('UNKNOWN', inplace=True) # 以防万一，进行常规 fillna
            cleaned_df.loc[cleaned_df['CustomerID'] == '', 'CustomerID'] = 'UNKNOWN' # 处理空字符串

            print(f"已处理 CustomerID 中的缺失值。对 {cleaned_df[cleaned_df['CustomerID'] == 'UNKNOWN'].shape[0]} 条目使用 'UNKNOWN'。")
            # 确保是字符串后，如果从浮点数转换而来，则移除 .0
            cleaned_df['CustomerID'] = cleaned_df['CustomerID'].apply(lambda x: x.split('.')[0] if isinstance(x, str) and '.' in x else x)
        else:
            print("警告: 未找到 'CustomerID' 列。")

        if 'Description' in cleaned_df.columns:
            cleaned_df['Description'] = cleaned_df['Description'].astype(str).str.strip()
            cleaned_df.dropna(subset=['Description'], inplace=True) # 通常 Description 为 NaN 表示该行为错误行
            print(f"丢弃空 Description 后的行数: {len(cleaned_df)}")

        if 'StockCode' in cleaned_df.columns:
            cleaned_df['StockCode'] = cleaned_df['StockCode'].astype(str).str.strip()
            cleaned_df.dropna(subset=['StockCode'], inplace=True)
            print(f"丢弃空 StockCode 后的行数: {len(cleaned_df)}")
        else:
            print("警告: 未找到 'StockCode' 列。这是一个关键标识符。")

        if 'Quantity' in cleaned_df.columns and 'UnitPrice' in cleaned_df.columns:
            print(f"Quantity/UnitPrice 特定清理前的初始行数: {len(cleaned_df)}")
            if remove_cancelled_orders:
                original_len = len(cleaned_df)
                cleaned_df = cleaned_df[cleaned_df['Quantity'] > 0]
                print(f"移除了 {original_len - len(cleaned_df)} 行 Quantity <= 0 的记录。")
            
            if remove_zero_unit_price:
                original_len = len(cleaned_df)
                cleaned_df = cleaned_df[cleaned_df['UnitPrice'] > 0]
                print(f"移除了 {original_len - len(cleaned_df)} 行 UnitPrice <= 0 的记录。")

            # 异常值处理应在移除已取消订单和零价格之后进行
            # 以避免这些已知的无效值扭曲 IQR 计算。
            if handle_outliers_quantity:
                cleaned_df = self._handle_column_outliers(cleaned_df, 'Quantity')
                
            
            if handle_outliers_unitprice:
                cleaned_df = self._handle_column_outliers(cleaned_df, 'UnitPrice')
        else:
            print("警告: 未找到 'Quantity' 或 'UnitPrice' 列。无法执行相关清理或异常值处理。")

        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        rows_dropped = initial_rows - len(cleaned_df)
        if rows_dropped > 0:
            print(f"丢弃了 {rows_dropped} 行重复记录。")

        print(f"数据清理完成。最终形状: {cleaned_df.shape}")
        self.processed_data = cleaned_df
        return self.processed_data

    """
            执行数据转换和特征工程。

            参数:
                df (pd.DataFrame): 要转换的 DataFrame (应为已清理数据)。

            返回:
                pd.DataFrame: 转换后的 DataFrame。
            """
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("开始数据转换...")
        transformed_df = df.copy()

        if 'Quantity' in transformed_df.columns and 'UnitPrice' in transformed_df.columns:
            transformed_df['TotalPrice'] = transformed_df['Quantity'] * transformed_df['UnitPrice']
            print("已创建 'TotalPrice' 列。")
        else:
            print("警告: 由于缺少 'Quantity' 或 'UnitPrice'，无法创建 'TotalPrice'。")

        if 'InvoiceDate' in transformed_df.columns and pd.api.types.is_datetime64_any_dtype(transformed_df['InvoiceDate']):
            transformed_df['OrderYear'] = transformed_df['InvoiceDate'].dt.year
            transformed_df['OrderMonth'] = transformed_df['InvoiceDate'].dt.month
            transformed_df['OrderDay'] = transformed_df['InvoiceDate'].dt.day
            transformed_df['OrderDayOfWeek'] = transformed_df['InvoiceDate'].dt.dayofweek # 星期一=0, 星期日=6
            transformed_df['OrderHour'] = transformed_df['InvoiceDate'].dt.hour
            transformed_df['OrderWeekOfYear'] = transformed_df['InvoiceDate'].dt.isocalendar().week.astype(int)
            transformed_df['OrderQuarter'] = transformed_df['InvoiceDate'].dt.quarter
            print("已提取日期特征: 年, 月, 日, 星期几, 小时, 年中的第几周, 季度。")
        else:
            print("警告: 未找到 'InvoiceDate' 列或非 datetime 类型。无法提取日期特征。")

        if 'InvoiceNo' in transformed_df.columns:
            transformed_df['InvoiceNo'] = transformed_df['InvoiceNo'].astype(str)
            transformed_df['IsCancelled'] = transformed_df['InvoiceNo'].str.startswith('C', na=False)
            print("已根据 'InvoiceNo' 创建 'IsCancelled' 列。")
        else:
            print("警告: 未找到 'InvoiceNo' 列。无法创建 'IsCancelled' 特征。")

        # 如果 CustomerID 存在，确保其为字符串类型，以保持一致性（已在 clean_data 中处理，但最好确保一下）
        if 'CustomerID' in transformed_df.columns:
            transformed_df['CustomerID'] = transformed_df['CustomerID'].astype(str)

        print(f"数据转换完成。最终形状: {transformed_df.shape}")
        self.processed_data = transformed_df
        return self.processed_data

    """
            将处理后的 DataFrame 保存到指定路径（例如，CSV 或 Excel）。

            参数:
                output_path (str): 保存处理后数据的路径。
            """
    def save_processed_data(self, output_path: str):
        if self.processed_data is not None:
            try:
                file_extension = output_path.split('.')[-1].lower()
                if file_extension == 'csv':
                    self.processed_data.to_csv(output_path, index=False)
                elif file_extension in ['xls', 'xlsx']:
                    self.processed_data.to_excel(output_path, index=False)
                elif file_extension == 'parquet':
                    self.processed_data.to_parquet(output_path, index=False)
                else:
                    raise ValueError("不支持的输出文件格式。请使用 csv、xlsx 或 parquet。")
                print(f"处理后的数据已保存到 {output_path}")
            except Exception as e:
                print(f"将处理后的数据保存到 {output_path} 时出错: {e}")
        else:
            print("没有要保存的处理后数据。请先运行清理/转换。")

    def calculate_total_quantity_sold(self, df: pd.DataFrame, 
                                      product_id_col: str = 'StockCode', 
                                      quantity_col: str = 'Quantity') -> Optional[pd.DataFrame]:
        """
        计算每个产品的总销售量。
        假设 Quantity > 0 表示销售。

        参数:
            df (pd.DataFrame): 包含交易数据的DataFrame。
            product_id_col (str): 产品标识符列的名称 (例如 'StockCode', 'Description')。
            quantity_col (str): 数量列的名称。

        返回:
            pd.DataFrame: 包含 'product_id_col' 和 'TotalSold' 列的DataFrame，如果出错则返回None。
        """
        if product_id_col not in df.columns or quantity_col not in df.columns:
            print(f"错误: 列 '{product_id_col}' 或 '{quantity_col}' (或两者) 在DataFrame中未找到。")
            return None
        
        if not pd.api.types.is_numeric_dtype(df[quantity_col]):
            print(f"错误: 数量列 '{quantity_col}' 必须是数值类型。")
            return None
        
        print(f"正在计算每个 '{product_id_col}' 的总销售量 (基于 '{quantity_col}')...")
        try:
            # 我们只考虑 Quantity > 0 的作为销售
            # 并确保 product_id_col 不是空的或NA
            sales_df = df[(df[quantity_col] > 0) & (df[product_id_col].notna()) & (df[product_id_col] != '')].copy()
            
            if sales_df.empty:
                print(f"警告: 在 '{product_id_col}' 和 '{quantity_col}' (大于0) 中没有找到有效的销售数据。")
                return pd.DataFrame(columns=[product_id_col, 'TotalSold'])

            total_sold = sales_df.groupby(product_id_col)[quantity_col].sum().reset_index()
            total_sold.rename(columns={quantity_col: 'TotalSold'}, inplace=True)
            print(f"总销售量计算完成。发现了 {len(total_sold)} 种产品。")
            return total_sold
        except Exception as e:
            print(f"计算总销售量时出错: {e}")
            return None

# 示例用法（用于在此文件中进行测试或演示）
if __name__ == '__main__':
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, '../../'))
    raw_data_file = 'Online Retail.xlsx' 
    raw_data_path = os.path.join(project_root, 'data', 'raw', raw_data_file)
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True) # Ensure processed dir exists

    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}")
        print(f"Please ensure the '{raw_data_file}' dataset is placed in the 'data/raw' directory.")
    else:
        processor = DataProcessor(data_path=raw_data_path)
        raw_df = processor.load_data()

        if raw_df is not None:
            print("\n--- Running Cleaning (default: no outlier handling) ---")
            cleaned_df_default = processor.clean_data(raw_df.copy(), 
                                                      remove_cancelled_orders=True, 
                                                      remove_zero_unit_price=True,
                                                      handle_outliers_quantity=False, 
                                                      handle_outliers_unitprice=False)
            print("\nCleaned DataFrame (default) info:")
            cleaned_df_default.info(verbose=False, memory_usage='deep')
            print(f"\nNull values in cleaned data (default):\n{cleaned_df_default.isnull().sum().sum()} total nulls")

            print("\n--- Running Cleaning (with outlier handling for Quantity and UnitPrice) ---")
            cleaned_df_outliers = processor.clean_data(raw_df.copy(), 
                                                       remove_cancelled_orders=True, 
                                                       remove_zero_unit_price=True,
                                                       handle_outliers_quantity=True, 
                                                       handle_outliers_unitprice=True)
            print("\nCleaned DataFrame (with outlier handling) info:")
            cleaned_df_outliers.info(verbose=False, memory_usage='deep')
            print(f"\nNull values in cleaned data (with outlier handling):\n{cleaned_df_outliers.isnull().sum().sum()} total nulls")
            print("\nDescriptive stats for Quantity after outlier handling:")
            print(cleaned_df_outliers['Quantity'].describe())
            print("\nDescriptive stats for UnitPrice after outlier handling:")
            print(cleaned_df_outliers['UnitPrice'].describe())

            print("\n--- Running Transformation (on outlier handled data) ---")
            transformed_df = processor.transform_data(cleaned_df_outliers.copy())
            print("\nTransformed DataFrame info:")
            transformed_df.info(verbose=False, memory_usage='deep')
            print(f"\nNull values in transformed data:\n{transformed_df.isnull().sum().sum()} total nulls")
            print("\nTransformed DataFrame head:")
            print(transformed_df.head())
            
            # Save the fully processed data
            processor.save_processed_data(os.path.join(processed_data_dir, 'online_retail_transformed_example.parquet'))
            print("\nExample processing complete.") 
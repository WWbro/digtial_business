import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np # 为 plot_correlation_matrix 添加
from statsmodels.tsa.seasonal import seasonal_decompose

# 为中文字符配置 Matplotlib
# 这会尝试为 macOS 设置一个常用的中文字体。
# 如果 'PingFang SC' 不可用，用户可能需要调整 font_name。
# 也可以考虑使用 'Arial Unicode MS' 或 'SimHei' 等其他通用中文字体。
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Arial Unicode MS,只使用该字体，不使用其他字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    print(f"设置中文字体时出错: {e}。可能需要手动安装或指定一个可用的中文字体。")


class Plotter:
    """
    一个用于创建各种数据可视化图表的类。
    """

    def __init__(self, style='seaborn-v0_8-whitegrid'):
        """
        初始化 Plotter 并设置绘图样式。

        参数:
            style (str): 用于图表的 matplotlib/seaborn 样式。
        """
        try:
            plt.style.use(style)
            # 字体设置现在是全局的，但我们在这里重新应用它们
            # 以确保它们对此绘图器实例有效，
            # 特别是如果使用多个绘图器实例或配置。
            # 但如果已在上面全局设置，则此处直接重新应用 rcParams 可能多余。
            # 为稳健起见，让我们确保这些关键设置是激活的。
            plt.rcParams['font.sans-serif'] = plt.rcParams.get('font.sans-serif', ['PingFang SC', 'Arial Unicode MS', 'SimHei']) # 重新确认
            plt.rcParams['axes.unicode_minus'] = False
        except OSError:
            print(f"警告: 样式 '{style}' 未找到。使用默认样式。")
        except Exception as e:
            print(f"在 Plotter 初始化中应用样式或字体设置时出错: {e}")
        # sns.set_palette("husl") # 如果需要，可以为每个图表或全局设置调色板

    def plot_histogram(self, df: pd.DataFrame, column: str, title: str = None, 
                       xlabel: str = None, ylabel: str = '频数', bins: int = 30, 
                       ax: plt.Axes = None, save_path: str = None):
        """
        为 DataFrame 中的指定列绘制直方图。
        如果提供了 ax，则在该 Axes 上绘图。否则，创建新的 Figure/Axes。
        """
        if column not in df.columns:
            print(f"错误: 列 '{column}' 在 DataFrame 中未找到。")
            return

        _ax = ax
        if _ax is None:
            fig, _ax = plt.subplots(figsize=(10, 6))
        else:
            fig = _ax.figure # 从提供的坐标轴获取图形对象

        sns.histplot(df[column], bins=bins, kde=True, ax=_ax)
        _ax.set_title(title if title else f'{column} 的直方图')
        _ax.set_xlabel(xlabel if xlabel else column)
        _ax.set_ylabel(ylabel)
        
        if save_path and ax is None: # 仅当我们创建了图形时才保存
            try:
                fig.savefig(save_path)
                print(f"图表已保存到 {save_path}")
            except Exception as e:
                print(f"保存图表到 {save_path} 时出错: {e}")
            finally:
                plt.close(fig)
        elif ax is None: # 仅当我们创建了图形且没有 save_path 时才显示
            plt.show()
        # 如果提供了 ax，则调用者处理 fig.show() 或 fig.savefig()
        return _ax.figure # 返回图形对象，可用于 Streamlit 或进一步操作

    def plot_scatterplot(self, df: pd.DataFrame, x_column: str, y_column: str, 
                         hue_column: str = None, title: str = None, xlabel: str = None, 
                         ylabel: str = None, ax: plt.Axes = None, save_path: str = None):
        """
        绘制散点图。如果提供了 ax，则在该 Axes 上绘图。
        """
        if x_column not in df.columns or y_column not in df.columns:
            print(f"错误: 列 '{x_column}' 或 '{y_column}' (或两者) 未找到。")
            return
        if hue_column and hue_column not in df.columns:
            print(f"警告: 色调列 '{hue_column}' 未找到。不使用色调进行绘图。")
            hue_column = None

        _ax = ax
        if _ax is None:
            fig, _ax = plt.subplots(figsize=(10, 6))
        else:
            fig = _ax.figure

        sns.scatterplot(x=df[x_column], y=df[y_column], hue=df[hue_column] if hue_column else None, ax=_ax)
        _ax.set_title(title if title else f'{y_column} vs {x_column} 的散点图')
        _ax.set_xlabel(xlabel if xlabel else x_column)
        _ax.set_ylabel(ylabel if ylabel else y_column)
        if hue_column:
            _ax.legend(title=hue_column)
            
        if save_path and ax is None:
            try:
                fig.savefig(save_path)
                print(f"图表已保存到 {save_path}")
            except Exception as e:
                print(f"保存图表到 {save_path} 时出错: {e}")
            finally:
                plt.close(fig)
        elif ax is None:
            plt.show()
        return _ax.figure

    def plot_lineplot(self, df: pd.DataFrame, x_column: str, y_column: str, 
                      hue_column: str = None, title: str = None, xlabel: str = None, 
                      ylabel: str = None, marker: str = 'o', ax: plt.Axes = None, save_path: str = None):
        """
        绘制折线图。如果提供了 ax，则在该 Axes 上绘图。
        """
        if x_column not in df.columns or y_column not in df.columns:
            print(f"错误: 列 '{x_column}' 或 '{y_column}' (或两者) 未找到。")
            return
        if hue_column and hue_column not in df.columns:
            print(f"警告: 色调列 '{hue_column}' 未找到。不使用色调进行绘图。")
            hue_column = None

        _ax = ax
        if _ax is None:
            fig, _ax = plt.subplots(figsize=(12, 6))
        else:
            fig = _ax.figure

        sns.lineplot(x=df[x_column], y=df[y_column], hue=df[hue_column] if hue_column else None, marker=marker, ax=_ax)
        _ax.set_title(title if title else f'{y_column} 关于 {x_column} 的折线图')
        _ax.set_xlabel(xlabel if xlabel else x_column)
        _ax.set_ylabel(ylabel if ylabel else y_column)
        if df[x_column].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df[x_column]):
            fig.autofmt_xdate() # 自动格式化 x 轴日期标签以获得更好的外观
            # _ax.tick_params(axis='x', rotation=45) # 另一种旋转标签的方式
        if hue_column:
            _ax.legend(title=hue_column)

        if save_path and ax is None:
            try:
                fig.savefig(save_path)
                print(f"图表已保存到 {save_path}")
            except Exception as e:
                print(f"保存图表到 {save_path} 时出错: {e}")
            finally:
                plt.close(fig)
        elif ax is None:
            plt.show()
        return _ax.figure

    def plot_correlation_matrix(self, df: pd.DataFrame, title: str = "相关系数矩阵", 
                                ax: plt.Axes = None, save_path: str = None):
        """
        绘制相关系数矩阵的热力图。如果提供了 ax，则在该 Axes 上绘图。
        """
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty or numeric_df.shape[1] < 2:
            print("错误: 找不到足够多的数值列用于计算相关系数矩阵 (至少需要2列)。")
            if ax is not None: # 如果传递了 ax，清除它并显示消息
                ax.clear()
                ax.text(0.5, 0.5, '没有足够的数值数据用于相关矩阵', ha='center', va='center')
                return ax.figure
            return None # 如果没有 ax，则无法返回图形对象
        
        correlation_matrix = numeric_df.corr()
        _ax = ax
        if _ax is None:
            fig, _ax = plt.subplots(figsize=(12, 10))
        else:
            fig = _ax.figure
            
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=_ax)
        _ax.set_title(title)
        
        if save_path and ax is None:
            try:
                fig.savefig(save_path)
                print(f"图表已保存到 {save_path}")
            except Exception as e:
                print(f"保存图表到 {save_path} 时出错: {e}")
            finally:
                plt.close(fig)
        elif ax is None:
            plt.show()
        return _ax.figure
    
    def plot_pie_chart(self, df: pd.DataFrame, values_column: str, names_column: str,
                       title: str = None, ax: plt.Axes = None, save_path: str = None, max_slices: int = 10):
        """
        绘制饼图。如果提供了 ax，则在该 Axes 上绘图。
        如果超出 max_slices，则将较小的切片分组为"其他"。
        """
        if values_column not in df.columns or names_column not in df.columns:
            print(f"错误: 列 '{values_column}' 或 '{names_column}' 未找到。")
            return

        _ax = ax
        if _ax is None:
            fig, _ax = plt.subplots(figsize=(10, 8))
        else:
            fig = _ax.figure

        # 为饼图聚合数据，例如，按 names_column 分组的 values_column 的总和。
        # 这假设 df 对于同一个名称可能有多行，所以我们进行聚合。
        # 如果 df 已经聚合，则此步骤可能不是必需的，但更安全。
        pie_data = df.groupby(names_column)[values_column].sum().sort_values(ascending=False)
        
        if len(pie_data) == 0:
            print(f"警告: 没有数据可用于绘制列 '{values_column}' 和 '{names_column}' 的饼图。")
            if ax is not None:
                ax.clear()
                ax.text(0.5, 0.5, '无数据显示', ha='center', va='center')
                return ax.figure
            return None


        if len(pie_data) > max_slices:
            top_slices = pie_data.iloc[:max_slices-1]
            others_sum = pie_data.iloc[max_slices-1:].sum()
            # 使用 pd.concat 替换直接赋值，以避免 SettingWithCopyWarning
            if others_sum > 0:
                 others_series = pd.Series([others_sum], index=['其他'])
                 pie_data_to_plot = pd.concat([top_slices, others_series])
            else:
                pie_data_to_plot = top_slices

        else:
            pie_data_to_plot = pie_data

        _ax.pie(pie_data_to_plot, labels=pie_data_to_plot.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.7))
        _ax.set_title(title if title else f'{values_column} 按 {names_column} 的饼图')
        _ax.axis('equal')  # 等宽高比确保饼图绘制为圆形。

        if save_path and ax is None:
            try:
                fig.savefig(save_path)
                print(f"图表已保存到 {save_path}")
            except Exception as e:
                print(f"保存图表到 {save_path} 时出错: {e}")
            finally:
                plt.close(fig)
        elif ax is None:
            plt.show()
        return _ax.figure

    def plot_bar_chart(self, df: pd.DataFrame, x_column: str, y_column: str, 
                       title: str = None, xlabel: str = None, ylabel: str = None, 
                       palette: str = "viridis", horizontal: bool = False,
                       ax: plt.Axes = None, save_path: str = None, top_n: int = None):
        """
        绘制条形图。如果提供了 ax，则在该 Axes 上绘图。
        可以根据 y_column 显示前 N 个类别。
        如果 x_column 有重复项，则假定 df 可能需要聚合，对 y_column 求和。
        """
        if x_column not in df.columns or y_column not in df.columns:
            print(f"错误: 列 '{x_column}' 或 '{y_column}' 未找到。")
            return

        _ax = ax
        if _ax is None:
            fig, _ax = plt.subplots(figsize=(12, 7) if not horizontal else (10, 8)) # 根据方向调整图形大小
        else:
            fig = _ax.figure

        # 聚合数据: 按 x_column 分组的 y_column 的总和
        bar_data = df.groupby(x_column, as_index=False)[y_column].sum()
        
        # 根据 y_column 的值排序，以便 top_n 生效
        sort_order = not horizontal # 垂直图表降序，水平图表升序（为了显示效果）
        bar_data = bar_data.sort_values(by=y_column, ascending=not sort_order)


        if top_n and top_n > 0 and top_n < len(bar_data):
            bar_data = bar_data.head(top_n)

        if horizontal:
            sns.barplot(x=y_column, y=x_column, data=bar_data, palette=palette, ax=_ax, orient='h')
            _ax.set_xlabel(ylabel if ylabel else y_column) # x 轴现在是值
            _ax.set_ylabel(xlabel if xlabel else x_column) # y 轴现在是类别
        else:
            sns.barplot(x=x_column, y=y_column, data=bar_data, palette=palette, ax=_ax)
            _ax.set_xlabel(xlabel if xlabel else x_column)
            _ax.set_ylabel(ylabel if ylabel else y_column)
            _ax.tick_params(axis='x', rotation=45) # 旋转x轴标签以便更好地显示

        _ax.set_title(title if title else f'{y_column} 按 {x_column} 的条形图{" (前 " + str(top_n) + " 名)" if top_n else ""}')
        
        plt.tight_layout() # 调整布局以防止标签重叠

        if save_path and ax is None:
            try:
                fig.savefig(save_path)
                print(f"图表已保存到 {save_path}")
            except Exception as e:
                print(f"保存图表到 {save_path} 时出错: {e}")
            finally:
                plt.close(fig)
        elif ax is None:
            plt.show()
        return _ax.figure

    def plot_sales_trend(self, df: pd.DataFrame, date_column: str, sales_column: str, 
                         title: str = None, xlabel: str = None, ylabel: str = None, 
                         ax: plt.Axes = None, save_path: str = None):
        """
        绘制销量随时间变化的趋势图 (折线图)。
        如果提供了 ax，则在该 Axes 上绘图。
        """
        if date_column not in df.columns or sales_column not in df.columns:
            print(f"错误: 列 '{date_column}' 或 '{sales_column}' 未找到。")
            if ax: # 如果有ax，清空并提示
                ax.clear()
                ax.text(0.5, 0.5, '日期或销量列未找到', ha='center', va='center')
                return ax.figure
            return None

        _ax = ax
        if _ax is None:
            fig, _ax = plt.subplots(figsize=(12, 6))
        else:
            fig = _ax.figure

        # 确保数据按日期排序
        plot_df_trend = df.sort_values(by=date_column)

        sns.lineplot(x=plot_df_trend[date_column], y=plot_df_trend[sales_column], marker='o', ax=_ax)
        _ax.set_title(title if title else f'{sales_column} 随 {date_column} 的趋势')
        _ax.set_xlabel(xlabel if xlabel else date_column)
        _ax.set_ylabel(ylabel if ylabel else sales_column)
        
        if pd.api.types.is_datetime64_any_dtype(plot_df_trend[date_column]):
            fig.autofmt_xdate()

        if save_path and ax is None:
            try:
                fig.savefig(save_path)
                print(f"图表已保存到 {save_path}")
            except Exception as e:
                print(f"保存图表到 {save_path} 时出错: {e}")
            finally:
                plt.close(fig)
        elif ax is None:
            plt.show()
        return _ax.figure

    def plot_time_series_decomposition(self, df: pd.DataFrame, date_column: str, value_column: str, 
                                       model: str = 'additive', period: int = None, 
                                       title: str = None, save_path: str = None):
        """
        绘制时间序列分解图 (趋势、季节性、残差)。
        需要用户指定季节性周期 (period)。
        """
        if date_column not in df.columns or value_column not in df.columns:
            print(f"错误: 日期列 '{date_column}' 或数值列 '{value_column}' 未找到。")
            return None # 改为返回None，让调用处可以检查

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            print(f"错误: 列 '{date_column}' 不是日期时间类型。")
            return None

        # 早期检查：确保 value_column 是数值类型
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            print(f"错误: 时间序列分解的目标列 '{value_column}' 必须是数值类型。检测到非数值数据。")
            # 可以在这里尝试找出第一个非数值项以提供更具体的错误，但这可能影响性能
            # non_numeric_example = df[value_column][~df[value_column].apply(lambda x: isinstance(x, (int, float, np.number)))].iloc[0]
            # print(f"例如，值 '{non_numeric_example}' 无法转换为数值。")
            return None

        # 准备数据：设置日期为索引，并确保数据是Series类型，按索引排序
        # 尝试在转换为 Series 之前处理潜在的非数值数据，或者确保它已经是数值的
        try:
            ts_data = df.set_index(date_column)[value_column].sort_index().astype(float) # 确保是 float 类型
        except ValueError as e:
            print(f"错误: 在为时间序列分解准备数据时，将列 '{value_column}' 转换为浮点数失败: {e}。请确保该列只包含可转换为数字的值。")
            return None

        # 尝试推断频率，如果推断不出且数据量较大，提示用户聚合或指定频率
        inferred_freq = pd.infer_freq(ts_data.index)
        if inferred_freq is None:
            if len(ts_data) > 200: # 对于较长序列，频率不确定性更大
                 print(f"警告: 无法从列 '{date_column}' 的索引中推断出频率。分解可能不准确或失败。请考虑将数据重采样到固定频率 (例如 'D' for daily, 'MS' for month start)。")
            # 即使频率未知，如果数据点足够，仍可以尝试分解，但用户提供的 period 非常重要
        else:
            print(f"从索引推断出的频率: {inferred_freq}")
            # 如果可以推断频率，可以基于频率提供默认周期建议，但仍以用户输入为准
            if period is None: # 如果用户没给周期
                if 'M' in inferred_freq.upper(): period = 12
                elif 'W' in inferred_freq.upper(): period = 52 # 或 7 (如果看周内)
                elif 'Q' in inferred_freq.upper(): period = 4
                elif 'D' in inferred_freq.upper(): period = 7 # 默认按周看季节性
                else:
                    print(f"警告: 未能根据推断的频率 '{inferred_freq}' 自动设置合理的季节性周期。请手动指定 'period' 参数。")
                    return None


        if period is None or period < 2:
            print("错误: 时间序列分解需要一个有效的季节性周期 (period >= 2)。")
            # 可以在调用此函数前，在Streamlit界面强制用户输入一个合理的period
            return None

        if len(ts_data) < 2 * period:
            print(f"错误: 数据点数量 ({len(ts_data)}) 不足以进行周期为 {period} 的时间序列分解。至少需要 {2 * period} 个数据点。")
            return None
        
        try:
            # 执行分解
            decomposition = seasonal_decompose(ts_data, model=model, period=period, extrapolate_trend='freq')

            fig_decompose, (ax_orig, ax_trend, ax_seas, ax_resid) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            
            decomposition.observed.plot(ax=ax_orig, legend=False)
            ax_orig.set_ylabel('原始数据')
            
            decomposition.trend.plot(ax=ax_trend, legend=False)
            ax_trend.set_ylabel('趋势')
            
            decomposition.seasonal.plot(ax=ax_seas, legend=False)
            ax_seas.set_ylabel('季节性')
            
            decomposition.resid.plot(ax=ax_resid, legend=False)
            ax_resid.set_ylabel('残差')
            
            fig_decompose.suptitle(title if title else f"{value_column} 的时间序列分解 (周期={period})", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以防与主标题重叠

            if save_path:
                try:
                    fig_decompose.savefig(save_path)
                    print(f"分解图已保存到 {save_path}")
                except Exception as e:
                    print(f"保存分解图到 {save_path} 时出错: {e}")
                finally:
                    plt.close(fig_decompose) # 保存后关闭
            else:
                # 在Streamlit中，我们通常返回fig对象让Streamlit的pyplot显示
                # plt.show() # 这在脚本中会阻塞，不适合Streamlit直接调用
                pass # 返回fig对象
            
            return fig_decompose

        except ValueError as ve:
            print(f"时间序列分解时发生 ValueError: {ve}。通常这意味着周期选择不当、数据包含过多NaN或数据频率问题。")
            return None
        except Exception as e:
            print(f"时间序列分解时发生未知错误: {e}")
            return None

# 示例用法
if __name__ == '__main__':
    # 创建一些示例数据
    data = {
        'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'D', 'A', 'B', 'C', 'D', 'E', 'E', 'A'],
        'Value1': np.random.rand(15) * 100,
        'Value2': np.random.rand(15) * 50,
        'Date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22', '2023-01-29',
                                '2023-02-05', '2023-02-12', '2023-02-19', '2023-02-26', '2023-03-05',
                                '2023-03-12', '2023-03-19', '2023-03-26', '2023-04-02', '2023-04-09']),
        'Group': np.random.choice(['X', 'Y'], size=15)
    }
    sample_df = pd.DataFrame(data)
    sample_df['Value1'] = sample_df['Value1'].astype(int)
    sample_df['Value2'] = sample_df['Value2'].astype(int)

    plotter = Plotter()

    print("正在绘制直方图...")
    plotter.plot_histogram(sample_df, 'Value1', title='Value1 的分布情况', xlabel='值', ylabel='数量')
    
    # 创建一个包含多个子图的图形
    fig_multi, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig_multi.suptitle("多图表展示", fontsize=16)


    print("\n正在绘制散点图 (在子图上)...")
    plotter.plot_scatterplot(sample_df, 'Value1', 'Value2', hue_column='Category', 
                             title='Value2 vs Value1 (按类别)', ax=axes[0, 0])

    print("\n正在绘制折线图 (在子图上)...")
    # 为了折线图，我们按日期排序并聚合，假设我们要看Value1随时间的变化
    line_data = sample_df.sort_values('Date').groupby('Date')['Value1'].sum().reset_index()
    plotter.plot_lineplot(line_data, 'Date', 'Value1', 
                          title='Value1 随时间的变化', ax=axes[0, 1])

    print("\n正在绘制相关系数矩阵 (在子图上)...")
    plotter.plot_correlation_matrix(sample_df[['Value1', 'Value2']], ax=axes[1, 0])
    
    print("\n正在绘制饼图 (在子图上)...")
    plotter.plot_pie_chart(sample_df, 'Value1', 'Category', title='Value1 按类别的饼图', ax=axes[1, 1], max_slices=5)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) #调整整体布局以适应主标题
    plt.show() # 显示包含所有子图的图形

    print("\n正在绘制单独的条形图 (垂直)...")
    plotter.plot_bar_chart(sample_df, 'Category', 'Value1', title='Value1 按类别的条形图 (前3名)', top_n=3)

    print("\n正在绘制单独的条形图 (水平)...")
    plotter.plot_bar_chart(sample_df, 'Category', 'Value1', title='Value1 按类别的水平条形图', horizontal=True, palette='magma', top_n=4)

    # 测试保存功能
    # output_dir = "../../reports/figures" # 假设的输出目录
    # import os
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # print(f"\n尝试将图表保存到 {output_dir} (如果取消注释代码)")
    # plotter.plot_histogram(sample_df, 'Value2', save_path=f"{output_dir}/histogram_value2.png")
    # plotter.plot_scatterplot(sample_df, 'Value1', 'Value2', save_path=f"{output_dir}/scatter_v1_v2.png")
    
    print("\n可视化示例完成。")
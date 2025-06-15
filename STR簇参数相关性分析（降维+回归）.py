import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from tkinter import filedialog
import tkinter as tk

def main():
    # 隐藏 Tkinter 主窗口
    root = tk.Tk()
    root.withdraw()

    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(filetypes=[('Excel files', '*.xlsx')])

    if not file_path:
        print("未选择文件。")
        return

    # 读取数据
    data = pd.read_excel(file_path, header=None)

    # 获取 Groups
    groups = data.iloc[0, 1:].values  # 排除第一列（字符串型列标题）

    # 获取参数名称
    params = data.iloc[1, 1:].values  # 排除第一列

    # 组合 Groups 和 Parameters 作为列名
    col_names = [f"{group}_{param}" for group, param in zip(groups, params)]

    # 创建测量值 DataFrame
    measurements = data.iloc[2:, 1:]
    measurements_df = pd.DataFrame(measurements.values, columns=col_names, index=data.iloc[2:, 0])

    # 将数据转换为数值类型
    measurements_df = measurements_df.apply(pd.to_numeric, errors='coerce')

    # 删除所有值都为 NaN 的行
    measurements_df = measurements_df.dropna(how='all')

    # 检查数据是否为空
    if measurements_df.empty:
        print("所有数据均为空，请检查输入文件。")
        return

    # 保留原始顺序获取唯一的 Groups
    unique_groups = pd.unique(groups)

    # 按组提取数据
    group_data = {}
    for group in unique_groups:
        group_cols = [col for col in measurements_df.columns if col.startswith(str(group))]
        group_df = measurements_df[group_cols].dropna()
        if group_df.empty:
            print(f"组 {group} 的数据为空，跳过该组。")
            continue
        group_data[group] = group_df

    # 如果没有有效的组数据，退出程序
    if not group_data:
        print("没有有效的组数据可供分析。")
        return

    # 定义降维方法
    dr_methods = {
        'PCA': PCA(n_components=2),
        'KernelPCA': KernelPCA(n_components=2, kernel='rbf')
    }

    # 定义回归方法
    reg_methods = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }

    # 存储组间分析的最佳结果
    best_avg_r2 = -np.inf
    best_corr_matrix = None
    best_dr_method_name = ''
    best_reg_method_name = ''
    best_group_dr_data = None  # 用于存储最佳降维后的数据

    # 组间分析
    for dr_name, dr_model in dr_methods.items():
        # 对每个组的数据进行降维
        group_dr_data = {}
        for group in group_data.keys():
            # 标准化数据
            scaler = StandardScaler()
            group_scaled = scaler.fit_transform(group_data[group])

            # 检查是否有样本
            if group_scaled.shape[0] == 0:
                print(f"组 {group} 在标准化后没有样本，跳过该组。")
                continue

            # 降维
            group_dr = dr_model.fit_transform(group_scaled)
            group_dr_data[group] = group_dr

        if len(group_dr_data) < 2:
            print("有效组数量不足以进行组间分析。")
            continue

        for reg_name, reg_model in reg_methods.items():
            from itertools import combinations

            group_pairs = combinations(group_dr_data.keys(), 2)

            # 初始化相关性系数矩阵
            n_groups = len(unique_groups)
            corr_matrix = np.zeros((n_groups, n_groups))

            group_indices = {group: idx for idx, group in enumerate(unique_groups)}

            r2_values = []

            for g1, g2 in group_pairs:
                # 获取两个组的降维数据
                data1 = group_dr_data[g1]  # 形状 (n_samples, n_components)
                data2 = group_dr_data[g2]  # 形状 (n_samples, n_components)

                # 确保样本数量匹配
                min_samples = min(data1.shape[0], data2.shape[0])
                if min_samples == 0:
                    print(f"组 {g1} 和组 {g2} 的样本数量不足，跳过该对。")
                    continue
                data1 = data1[:min_samples]
                data2 = data2[:min_samples]

                # data1 是 X，data2 是 y
                # 拟合回归模型
                model = reg_model
                model.fit(data1, data2)

                # 计算 R² 值
                r_squared = model.score(data1, data2)
                r2_values.append(r_squared)

                # 存储 R² 值
                idx1 = group_indices[g1]
                idx2 = group_indices[g2]
                corr_matrix[idx1, idx2] = r_squared
                corr_matrix[idx2, idx1] = r_squared  # 对称矩阵

            if not r2_values:
                print("没有有效的 R² 值计算，跳过当前配置。")
                continue

            # 对角线赋值为 1
            np.fill_diagonal(corr_matrix, 1)

            # 计算平均 R² 值
            avg_r2 = np.mean(r2_values)

            # 更新最佳结果
            if avg_r2 > best_avg_r2:
                best_avg_r2 = avg_r2
                best_corr_matrix = corr_matrix.copy()
                best_dr_method_name = dr_name
                best_reg_method_name = reg_name
                # 保存最佳的降维数据
                best_group_dr_data = group_dr_data.copy()

    if best_avg_r2 == -np.inf:
        print("无法找到最佳的组间分析结果。")
        return

    # 组内参数相关性分析
    # 存储结果的字典
    group_corr_matrices = {}
    for group in group_data.keys():
        # 获取组数据
        data = group_data[group]
        params = data.columns  # 参数名称

        # 对每个参数的测量值进行降维
        param_reduced_data = {}
        for param in params:
            param_data = data[[param]]

            # 检查是否有样本
            if param_data.shape[0] == 0:
                print(f"参数 {param} 在组 {group} 中没有样本，跳过该参数。")
                continue

            # 标准化处理
            scaler = StandardScaler()
            param_scaled = scaler.fit_transform(param_data)
            param_reduced_data[param] = param_scaled

        n_params = len(param_reduced_data)
        if n_params < 2:
            print(f"组 {group} 中有效参数不足，跳过组内相关性分析。")
            continue

        # 计算参数之间的相关性系数
        corr_matrix = np.zeros((n_params, n_params))

        param_list = list(param_reduced_data.keys())
        param_indices = {param: idx for idx, param in enumerate(param_list)}

        for i, param_i in enumerate(param_list):
            for j in range(i + 1, n_params):
                param_j = param_list[j]
                data_i = param_reduced_data[param_i]
                data_j = param_reduced_data[param_j]

                # 确保样本数量匹配
                min_samples = min(data_i.shape[0], data_j.shape[0])
                if min_samples == 0:
                    print(f"参数 {param_i} 和参数 {param_j} 的样本数量不足，跳过该对。")
                    continue
                data_i = data_i[:min_samples]
                data_j = data_j[:min_samples]

                # data_i 是 X，data_j 是 y
                # 拟合回归模型
                model = reg_methods[best_reg_method_name]
                model.fit(data_i, data_j)

                # 计算 R² 值
                r_squared = model.score(data_i, data_j)

                # 存储 R² 值
                idx_i = param_indices[param_i]
                idx_j = param_indices[param_j]
                corr_matrix[idx_i, idx_j] = r_squared
                corr_matrix[idx_j, idx_i] = r_squared  # 对称矩阵

        # 对角线赋值为 1
        np.fill_diagonal(corr_matrix, 1)

        # 创建相关性矩阵的 DataFrame
        corr_df = pd.DataFrame(corr_matrix, index=param_list, columns=param_list)
        group_corr_matrices[group] = corr_df

    # 输出最佳组间相关性系数矩阵
    between_corr_df = pd.DataFrame(best_corr_matrix, index=unique_groups, columns=unique_groups)

    print(f"最佳降维方法：{best_dr_method_name}")
    print(f"最佳回归方法：{best_reg_method_name}")
    print(f"组间平均 R² 值：{best_avg_r2}")

    # 使用最佳降维方法重新降维并导出数据
    # 对每个组的数据进行降维
    best_dr_model = dr_methods[best_dr_method_name]

    # 字典来存储降维后的数据 DataFrame
    best_group_dr_dfs = {}

    for group in group_data.keys():
        # 标准化数据
        scaler = StandardScaler()
        group_scaled = scaler.fit_transform(group_data[group])

        # 检查是否有样本
        if group_scaled.shape[0] == 0:
            print(f"组 {group} 在标准化后没有样本，跳过该组的降维。")
            continue

        # 降维
        group_dr = best_dr_model.fit_transform(group_scaled)
        # 创建 DataFrame
        n_components = group_dr.shape[1]
        component_names = [f"Component_{i+1}" for i in range(n_components)]
        group_dr_df = pd.DataFrame(group_dr, columns=component_names, index=group_data[group].index)
        best_group_dr_dfs[group] = group_dr_df

    if not best_group_dr_dfs:
        print("没有降维后的数据可供保存。")
        return

    # 将相关性矩阵和降维后的数据保存为含多个 Sheet 的 Excel 文件
    output_file = 'correlation_matrices.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        # 写入组间相关性矩阵
        between_corr_df.to_excel(writer, sheet_name='Between_Groups')

        # 写入组内相关性矩阵
        for group, corr_df in group_corr_matrices.items():
            corr_df.to_excel(writer, sheet_name=f'Within_{group}')

        # 写入降维后的数据
        for group, dr_df in best_group_dr_dfs.items():
            dr_df.to_excel(writer, sheet_name=f'DR_{group}')

    print(f"相关性矩阵和降维后的数据已保存到 {output_file}")

if __name__ == "__main__":
    main()

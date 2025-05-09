import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from experiment.masg import LDPGroupBy_masg


# 对value_normalized进行离散化
def discretize_value(v):
    v = np.clip(v, -1, 1)
    prob = (1 + v) / 2
    return np.random.choice([1, -1], p=[prob, 1 - prob])


def iterative_sample_allocation(restored_mean, restored_variance, total_budget, key_space, n_key_1, epsilon, max_iter=100,
                                tol=1):
    """
    采用多轮迭代计算各组样本量 s_i, 满足 sum_i(s_i) = total_budget。
    """
    d = len(key_space)

    epsilon_1 = np.log((np.exp(epsilon) + 1) / 2)
    epsilon_2 = epsilon
    # key扰动的概率
    a = np.exp(epsilon_1) / (np.exp(epsilon_1) + n_key_1 - 1)
    c = 1 / (np.exp(epsilon_1) + n_key_1 - 1)
    # value扰动概率
    p = np.exp(epsilon_2) / (np.exp(epsilon_2) + 1)
    q = 1 / (np.exp(epsilon_2) + 1)

    # 初始化 s_i^0 = total_budget / d
    s_i_dict = {key: (total_budget / d) for key in key_space}

    for iteration in range(max_iter):
        # 计算 r_i = s_i / s
        r_i_dict = {key: s_i_dict[key] / total_budget for key in key_space}

        partial_dict = {}
        T2_values = {}
        T3_values = {}
        noise_terms = {}
        for key in key_space:
            M_i = restored_mean[key]
            var_i = restored_variance[key]
            r_i = r_i_dict[key]  # 当前迭代的 r_i

            T2 = M_i ** 2
            T2_values[key] = T2

            T3 = ((c / r_i) + (a - c)) / (a ** 2 * (2 * p - 1) ** 2)
            T3_values[key] = T3
            noise_terms[key] = T3 - T2

            sum_inside = var_i - T2 + T3

            partial_dict[key] = np.sqrt(sum_inside) / M_i

        sum_partial = sum(partial_dict.values())

        # 更新 s_i
        new_s_i_dict = {}
        for key in key_space:
            new_s_i_dict[key] = total_budget * (partial_dict[key] / sum_partial)

        # 检查是否收敛
        max_diff = max(abs(new_s_i_dict[key] - s_i_dict[key]) for key in key_space)
        s_i_dict = new_s_i_dict

        if max_diff < tol:
            print(f"迭代在第 {iteration + 1} 轮收敛。")
            break

    return s_i_dict


if __name__ == '__main__':
    # 读取 CSV 文件
    file_path = "../../dataset/retirement/retirement_expanded.csv"
    table = pd.read_csv(file_path)  # nrows指定读取行数

    # 定义分组查询所需参数
    group_by_attributes = ['Organization Group Code']  # 替换为实际的分组列名
    aggregation_attribute_1 = 'Retirement'  # 替换为实际的聚合列名
    aggregation_attribute_2 = 'Total Benefits'  # 替换为实际的聚合列名
    memory_budget = 127348  # 总样本容量= 2% * 用户总数
    alpha = 0.2  # phase1 和 phase2 分配的样本数比例

    # 计算phase1和phase2的样本数
    mb_1_1 = int(memory_budget * alpha / 2)
    mb_1_2 = int(memory_budget * alpha / 2)
    mb_2 = int(memory_budget * (1 - alpha))

    # 定义差分隐私参数[2, 3, 4, 5, 6]
    epsilon = 2

    table_simple = table[[*group_by_attributes, aggregation_attribute_1, aggregation_attribute_2]].copy()
    table_simple = table_simple.dropna()
    for group_attr in group_by_attributes:
        table_simple[group_attr] = table_simple[group_attr].astype(str)

    total_users = len(table_simple)

    value_max_1 = table_simple[aggregation_attribute_1].max()
    value_min_1 = table_simple[aggregation_attribute_1].min()
    value_max_2 = table_simple[aggregation_attribute_2].max()
    value_min_2 = table_simple[aggregation_attribute_2].min()

    """在N1中均匀采样mb_1的数据量"""
    N1_agg1, N1_agg2, N2, N1_sample_size_1, N1_sample_size_2 = LDPGroupBy_masg.uniform_sampling(table_simple,
                                                                                                group_by_attributes[0],
                                                                                                aggregation_attribute_1,
                                                                                                aggregation_attribute_2,
                                                                                                mb_1_1, mb_1_2)

    """N1数据预处理"""
    N1_agg2[group_by_attributes[0]] = N1_agg2[group_by_attributes[0]].apply(lambda x: str(x) + '_2')

    # 归一化
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    scaler_2 = MinMaxScaler(feature_range=(0, 1))
    N1_agg1['value_normalized_1'] = scaler_1.fit_transform(N1_agg1[[aggregation_attribute_1]])
    N1_agg2['value_normalized_2'] = scaler_2.fit_transform(N1_agg2[[aggregation_attribute_2]])

    """加噪后CV"""
    key_space_1 = table_simple[group_by_attributes[0]].unique().tolist()
    n_key_1 = len(key_space_1)
    key_space_2 = N1_agg2[group_by_attributes[0]].unique().tolist()
    n_key_2 = len(key_space_2)
    key_space = key_space_1 + key_space_2
    n_key = len(key_space)

    simple_sizes_list = []
    estimated_frequency_list = []

    data_1 = list(N1_agg1[group_by_attributes + ['value_normalized_1']].itertuples(index=False, name=None))
    data_2 = list(N1_agg2[group_by_attributes + ['value_normalized_2']].itertuples(index=False, name=None))

    attempts = 0
    while len(simple_sizes_list) < 10 and attempts < 2000:
        try:
            # 扰动数据
            estimated_frequencies_1, estimated_means_1, estimated_variances_1 = LDPGroupBy_masg.group_perturb_phase1(
                data_1, key_space_1, epsilon)
            estimated_frequencies_2, estimated_means_2, estimated_variances_2 = LDPGroupBy_masg.group_perturb_phase1(
                data_2, key_space_2, epsilon)

            # 检查方差估计中是否存在负值，由于浮点数精度问题，所以会存在负值方差
            if any(variance < 0 for variance in estimated_variances_1.values()) or any(
                    variance < 0 for variance in estimated_variances_2.values()):
                attempts += 1
                continue

            restored_means_1 = {key: scaler_1.inverse_transform([[m_k]]).item() for key, m_k
                                in estimated_means_1.items()}
            restored_variances_1 = {key: variance * ((value_max_1 - value_min_1) ** 2) for key, variance in
                                    estimated_variances_1.items()}
            restored_means_2 = {key: scaler_2.inverse_transform([[m_k]]).item() for key, m_k
                                in estimated_means_2.items()}
            restored_variances_2 = {key: variance * ((value_max_2 - value_min_2) ** 2) for key, variance in
                                    estimated_variances_2.items()}

            estimated_means = {}
            for key in key_space_1:
                estimated_means[key] = estimated_means_1[key]
            for key in key_space_2:
                estimated_means[key] = estimated_means_2[key]

            estimated_variances = {}
            for key in key_space_1:
                estimated_variances[key] = estimated_variances_1[key]
            for key in key_space_2:
                estimated_variances[key] = estimated_variances_2[key]

            estimated_frequencies = {}
            for key in key_space_1:
                estimated_frequencies[key] = estimated_frequencies_1[key]
            for key in key_space_2:
                estimated_frequencies[key] = estimated_frequencies_2[key]

            # 迭代计算各组的采样量（不需要使用逆归一化均值和方差）
            simple_size = iterative_sample_allocation(estimated_means, estimated_variances, mb_2, key_space, n_key_1, epsilon)

            if all(value > 0 for value in simple_size.values()):
                simple_sizes_list.append(simple_size)
                estimated_frequency_list.append(estimated_frequencies)
            else:
                print(f"第{len(simple_sizes_list) + 1}次扰动结果无效，各分组样本量包含负值，跳过")

            attempts += 1
        except Exception as e:
            print(f"第{len(simple_sizes_list) + 1}次加噪后各分组样本量计算失败: {e}")
            attempts += 1

    average_simple_sizes = {}
    for key in simple_sizes_list[0].keys():
        average_simple_sizes[key] = np.mean([gamma_values[key] for gamma_values in simple_sizes_list])
    print("各分组平均样本量:", average_simple_sizes)

    average_estimated_frequency = {}
    for key in estimated_frequency_list[0].keys():
        average_estimated_frequency[key] = np.mean(
            [estimated_frequencies[key] for estimated_frequencies in estimated_frequency_list])

    """阶段2"""
    """采样样本"""
    simple_size_agg1 = {}
    simple_size_agg2 = {}
    for key in key_space_1:
        simple_size_agg1[key] = average_simple_sizes[key]

    for key in key_space_2:
        simple_size_agg2[key] = average_simple_sizes[key]

    simple_size_agg2_original = {}
    for key in key_space_2:
        original_key = key.replace('_2', '')
        simple_size_agg2_original[original_key] = simple_size_agg2[key]
    simple_size_agg2 = simple_size_agg2_original

    # 使用分层抽样
    sample_2_agg1, sample_2_agg2 = LDPGroupBy_masg.stratified_sampling_probability_optimized(
        N2,
        simple_size_agg1,
        simple_size_agg2,
        group_by_attributes[0]
    )

    """对样本数据添加噪声"""
    # 进行归一化和离散化
    scaler_3 = MinMaxScaler(feature_range=(-1, 1))
    scaler_4 = MinMaxScaler(feature_range=(-1, 1))

    sample_2_agg1 = sample_2_agg1.copy()
    sample_2_agg2 = sample_2_agg2.copy()

    sample_2_agg1.loc[:, 'value_normalized_1'] = scaler_3.fit_transform(sample_2_agg1[[aggregation_attribute_1]])
    sample_2_agg1.loc[:, 'value_discretized_1'] = sample_2_agg1['value_normalized_1'].apply(discretize_value)

    sample_2_agg2.loc[:, 'value_normalized_2'] = scaler_4.fit_transform(sample_2_agg2[[aggregation_attribute_2]])
    sample_2_agg2.loc[:, 'value_discretized_2'] = sample_2_agg2['value_normalized_2'].apply(discretize_value)

    sample_data_agg1 = list(
        sample_2_agg1[group_by_attributes + ['value_discretized_1']].itertuples(index=False, name=None))
    sample_data_agg2 = list(
        sample_2_agg2[group_by_attributes + ['value_discretized_2']].itertuples(index=False, name=None))

    real_result_agg1 = table_simple.groupby(group_by_attributes)[aggregation_attribute_1].agg('mean')
    real_result_agg2 = table_simple.groupby(group_by_attributes)[aggregation_attribute_2].agg('mean')
    print(f"各分组真实均值：agg1={real_result_agg1}, agg2={real_result_agg2}")

    relative_errors_list_agg1 = []
    relative_errors_list_agg2 = []
    sample_means_list_agg1 = []
    sample_means_list_agg2 = []

    for i in range(10):
        try:
            sample_estimated_frequencies_agg1, sample_estimated_means_agg1 = LDPGroupBy_masg.group_perturb_phase2(
                sample_data_agg1, key_space_1, epsilon)
            sample_estimated_frequencies_agg2, sample_estimated_means_agg2 = LDPGroupBy_masg.group_perturb_phase2(
                sample_data_agg2, key_space_1, epsilon)

            # 使用逆归一化公式还原均值和方差到原始数据大小
            sample_restored_means_agg1 = {key: scaler_3.inverse_transform([[m_k]]).item() for key, m_k
                                          in sample_estimated_means_agg1.items()}
            sample_restored_means_agg2 = {key: scaler_4.inverse_transform([[m_k]]).item() for key, m_k
                                          in sample_estimated_means_agg2.items()}

            sample_means_list_agg1.append(sample_restored_means_agg1)
            sample_means_list_agg2.append(sample_restored_means_agg2)

            sample_restored_means_series_agg1 = pd.Series(sample_restored_means_agg1)
            sample_restored_means_series_agg2 = pd.Series(sample_restored_means_agg2)
            relative_errors_agg1 = (
                        np.abs(sample_restored_means_series_agg1 - real_result_agg1) / real_result_agg1).to_dict()
            relative_errors_agg2 = (
                        np.abs(sample_restored_means_series_agg2 - real_result_agg2) / real_result_agg2).to_dict()
            relative_errors_list_agg1.append(relative_errors_agg1)
            relative_errors_list_agg2.append(relative_errors_agg2)

        except Exception as e:
            print(f"扰动样本数据失败: {e}")

    average_restored_means_agg1 = pd.DataFrame(sample_means_list_agg1).mean().to_dict()
    average_restored_means_agg2 = pd.DataFrame(sample_means_list_agg2).mean().to_dict()
    print("样本估计均值：agg1=", average_restored_means_agg1, ", agg2=", average_restored_means_agg2)

    average_relative_errors_agg1 = pd.DataFrame(relative_errors_list_agg1).mean().to_dict()
    average_relative_errors_agg2 = pd.DataFrame(relative_errors_list_agg2).mean().to_dict()
    print("相对误差：agg1=", average_relative_errors_agg1, ", agg2=", average_relative_errors_agg2)

    group_average_errors = {}
    for group in average_relative_errors_agg1.keys():
        group_average_errors[group] = (average_relative_errors_agg1[group] + average_relative_errors_agg2[group]) / 2
    print("每个组两个聚合属性的平均相对误差：", group_average_errors)

    average_all_relative_errors = np.mean(list(group_average_errors.values()))
    print("所有组的总体平均相对误差:", average_all_relative_errors)


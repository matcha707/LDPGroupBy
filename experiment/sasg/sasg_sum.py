import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from experiment.sasg import LDPGroupBy


# 对value_normalized进行离散化
def discretize_value(v):
    v = np.clip(v, -1, 1)
    prob = (1 + v) / 2
    return np.random.choice([1, -1], p=[prob, 1 - prob])


def iterative_sample_allocation(restored_mean, restored_variance, total_budget, key_space, epsilon, max_iter=100,
                                tol=1):
    """
    采用多轮迭代计算各组样本量 s_i, 满足 sum_i(s_i) = total_budget。
    """

    d = len(key_space)

    epsilon_1 = np.log((np.exp(epsilon) + 1) / 2)
    epsilon_2 = epsilon
    # key扰动的概率
    a = np.exp(epsilon_1) / (np.exp(epsilon_1) + d - 1)
    c = 1 / (np.exp(epsilon_1) + d - 1)
    # value扰动概率
    p = np.exp(epsilon_2) / (np.exp(epsilon_2) + 1)
    q = 1 / (np.exp(epsilon_2) + 1)

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
    # 读取 CSV 文件，需更换路径
    file_path = "../../dataset/retirement/retirement_expanded.csv"
    table = pd.read_csv(file_path)
    total_users = len(table)

    # 定义分组查询所需参数
    group_by_attributes = ['Organization Group Code']  # 替换为实际的分组列名
    aggregation_attribute = 'Retirement'  # 替换为实际的聚合列名
    memory_budget = 63674  # 总样本容量= 1% * 用户总数，替换为实际所需要的样本量
    alpha = 0.2  # phase1 和 phase2 分配的样本数比例，替换为所需要的alpha

    # 计算phase1和phase2的样本数
    mb_1 = int(memory_budget * alpha)
    mb_2 = int(memory_budget * (1 - alpha))

    # 隐私预算，替换为所需要的隐私预算[1, 2, 4, 6, 8]
    epsilon = 2

    table_simple = table[[*group_by_attributes, aggregation_attribute]].copy()
    table_simple = table_simple.dropna()

    value_max = table[aggregation_attribute].max()
    value_min = table[aggregation_attribute].min()

    """在N1中均匀采样mb_1的数据量"""
    N1, N2, sample_size_1 = LDPGroupBy.uniform_sampling(table_simple, group_by_attributes[0], mb_1)

    """N1数据预处理"""
    # 归一化
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    N1['value_normalized'] = scaler_1.fit_transform(N1[[aggregation_attribute]])

    """加噪后CV"""
    key_space = table_simple[group_by_attributes[0]].unique().tolist()
    n_key = len(key_space)

    simple_sizes_list = []
    estimated_frequency_list = []

    data = list(N1[group_by_attributes + ['value_normalized']].itertuples(index=False, name=None))

    attempts = 0
    while len(simple_sizes_list) < 10 and attempts < 2000:
        try:
            # 扰动数据
            estimated_frequencies, estimated_means, estimated_variances = LDPGroupBy.group_perturb_phase1(
                data, key_space, epsilon)

            # 检查方差估计中是否存在负值，由于浮点数精度问题，所以会存在负值方差
            if any(variance < 0 for variance in estimated_variances.values()):
                attempts += 1
                continue

            restored_means = {key: scaler_1.inverse_transform([[m_k]]).item() for key, m_k
                              in estimated_means.items()}
            restored_variances = {key: variance * ((value_max - value_min) ** 2) for key, variance in
                                  estimated_variances.items()}

            # 迭代计算各组的采样量（不需要使用逆归一化均值和方差）
            simple_size = iterative_sample_allocation(estimated_means, estimated_variances, mb_2, key_space, epsilon)

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
        average_estimated_frequency[key] = np.mean([estimated_frequencies[key] for estimated_frequencies in estimated_frequency_list])

    """阶段2"""
    """采样样本"""
    # 使用分层抽样
    sample_2, prob_dict = LDPGroupBy.stratified_sampling_probability(N2, average_simple_sizes, average_estimated_frequency, group_by_attributes[0])

    """对样本数据添加噪声"""
    # 进行归一化和离散化
    scaler_2 = MinMaxScaler(feature_range=(-1, 1))
    sample_2['value_normalized'] = scaler_2.fit_transform(sample_2[[aggregation_attribute]])
    sample_2['value_discretized'] = sample_2['value_normalized'].apply(discretize_value)

    sample_data = list(sample_2[group_by_attributes + ['value_discretized']].itertuples(index=False, name=None))

    real_result = table.groupby(group_by_attributes)[aggregation_attribute].sum()
    print(f"各分组真实总和：{real_result}")

    relative_errors_list = []
    sample_sum_list = []

    for i in range(10):
        try:
            sample_estimated_frequencies, sample_estimated_means = LDPGroupBy.group_perturb_phase2(
                sample_data, key_space, epsilon)

            sample_restored_means = {key: scaler_2.inverse_transform([[m_k]]).item() for key, m_k
                                     in sample_estimated_means.items()}

            # N1的count估计
            n1_count_estimate = {key: average_estimated_frequency[key] * total_users for key in key_space}

            # N2的count估计
            n2_count_estimate = {
                key: (sample_estimated_frequencies[key] * mb_2) / prob_dict[key] if prob_dict[key] > 0 else 0
                for key in key_space
            }

            sample_restored_counts = {
                key: (n1_count_estimate[key] + n2_count_estimate[key]) / 2
                for key in key_space
            }
            
            sample_restored_sum = {key: sample_restored_counts[key] * sample_restored_means[key] for key in key_space}

            sample_sum_list.append(sample_restored_sum)

            sample_restored_sum_series = pd.Series(sample_restored_sum)
            relative_errors = (np.abs(sample_restored_sum_series - real_result) / real_result).to_dict()
            relative_errors_list.append(relative_errors)

        except Exception as e:
            print(f"扰动样本数据失败: {e}")

    average_restored_sum = pd.DataFrame(sample_sum_list).mean().to_dict()
    print("估计总和：", average_restored_sum)

    average_relative_errors = pd.DataFrame(relative_errors_list).mean().to_dict()
    print("相对误差：", average_relative_errors)

    average_all_relative_errors = np.mean(list(average_relative_errors.values()))
    print("所有组的平均相对误差:", average_all_relative_errors)

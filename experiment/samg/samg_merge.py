from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from experiment.samg import LDPGroupBy_samg

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


def update_group_names(N2, merge_info, group_attribute):
    if not merge_info:
        return N2

    group_mapping = {}
    for merged_group, original_groups in merge_info.items():
        for orig_group in original_groups:
            group_mapping[orig_group] = merged_group

    N2_updated = N2.copy()

    N2_updated[group_attribute] = N2_updated[group_attribute].map(
        lambda x: group_mapping.get(x, x)
    )

    return N2_updated


def calculate_pairwise_distance(groups_means: dict, groups_variances: dict):
    """
    计算 small group 之间的均值和方差的欧式距离，并初始化 DependenceList 和 d_max
    """
    group_keys = list(groups_means.keys())
    num_groups = len(group_keys)
    distance_matrix = pd.DataFrame(np.zeros((num_groups, num_groups)), index=group_keys, columns=group_keys)
    dependence_list = []
    d_max = 0

    for i, key1 in enumerate(group_keys):
        for j, key2 in enumerate(group_keys):
            if i < j:
                mean_diff = (groups_means[key1] - groups_means[key2]) ** 2
                variance_diff = (groups_variances[key1] - groups_variances[key2]) ** 2
                # 计算欧式距离
                distance = np.sqrt(mean_diff + variance_diff)

                distance_matrix.at[key1, key2] = distance
                distance_matrix.at[key2, key1] = distance

                dependence_list.append((key1, key2, distance))
                d_max = max(d_max, distance)

    dependence_list.sort(key=lambda x: x[2])

    return distance_matrix, dependence_list, d_max


def similarity_based_group_merging(groups_frequencies: dict, groups_means: dict, groups_variances: dict,
                                   scaling_factor: float = 0.3, similar_threshold: float = 0.5):
    """
    基于相似度的组合并算法
    """
    merge_history = {}

    S = LDPGroupBy_samg.identify_small_groups(groups_frequencies, scaling_factor)

    # 初始化: G' ← G \ S
    G_prime = set(groups_frequencies.keys()) - set(S)

    if len(S) <= 1:
        G_prime.update(S)
        return groups_frequencies, groups_means, groups_variances, G_prime, merge_history

    current_frequencies = groups_frequencies.copy()
    current_means = groups_means.copy()
    current_variances = groups_variances.copy()

    num_groups = len(groups_frequencies)
    frequency_threshold = scaling_factor / num_groups

    while len(S) > 1:
        small_groups_means = {k: current_means[k] for k in S}
        small_groups_variances = {k: current_variances[k] for k in S}

        _, dependence_list, d_max = calculate_pairwise_distance(
            small_groups_means,
            small_groups_variances
        )

        if not dependence_list:
            break

        key1, key2, distance = dependence_list[0]
        d_norm = distance / d_max

        if d_norm >= similar_threshold:
            break

        merged_key = f"{key1}_{key2}"
        new_frequency, new_mean, new_variance = LDPGroupBy_samg.calculate_new_mean_and_variance(
            key1, key2,
            current_frequencies,
            current_means,
            current_variances
        )

        original_groups = []
        if key1 in merge_history:
            original_groups.extend(merge_history[key1])
            del merge_history[key1]
        else:
            original_groups.append(key1)

        if key2 in merge_history:
            original_groups.extend(merge_history[key2])
            del merge_history[key2]
        else:
            original_groups.append(key2)

        merge_history[merged_key] = original_groups

        # 根据合并组的频率更新S和G'
        S.remove(key1)
        S.remove(key2)

        # 判断新组的频率是否小于阈值
        if new_frequency < frequency_threshold:
            S.append(merged_key)
        else:
            G_prime.add(merged_key)

        # 更新合并组的频率，均值，方差
        current_frequencies[merged_key] = new_frequency
        current_means[merged_key] = new_mean
        current_variances[merged_key] = new_variance
        del current_frequencies[key1], current_frequencies[key2]
        del current_means[key1], current_means[key2]
        del current_variances[key1], current_variances[key2]

    G_prime.update(S)

    result_frequencies = {k: current_frequencies[k] for k in G_prime}
    result_means = {k: current_means[k] for k in G_prime}
    result_variances = {k: current_variances[k] for k in G_prime}

    return result_frequencies, result_means, result_variances, G_prime, merge_history


if __name__ == '__main__':
    file_path = "../../dataset/retirement/retirement_expanded.csv"
    table = pd.read_csv(file_path)
    total_users = len(table)

    # 定义分组查询所需参数
    group_by_attributes = ['Organization Group Code', 'Year Type']  # 替换为实际的分组列名
    aggregation_attribute = 'Retirement'  # 替换为实际的聚合列名
    memory_budget = 318370  # 总样本容量= 5% * 用户总数
    alpha = 0.4  # phase1 和 phase2 分配的样本数比例

    # 计算phase1和phase2的样本数
    mb_1 = int(memory_budget * alpha)
    mb_2 = int(memory_budget * (1 - alpha))

    # 定义差分隐私参数
    epsilon = 6

    table_simple = table[[*group_by_attributes, aggregation_attribute]].copy()
    table_simple = table_simple.dropna()
    group_attribute = 'group_key'
    table_simple[group_attribute] = table_simple[group_by_attributes[0]].astype(str).str.strip()
    for attr in group_by_attributes[1:]:
        table_simple[group_attribute] += '-' + table_simple[attr].astype(str).str.strip()
    unique_group_key = table_simple[group_attribute].unique().tolist()
    key_space = sorted(unique_group_key)
    n_key = len(key_space)

    value_max = table_simple[aggregation_attribute].max()
    value_min = table_simple[aggregation_attribute].min()

    """在N1中均匀采样mb_1的数据量"""
    N1, N2, sample_size_1 = LDPGroupBy_samg.uniform_sampling(table_simple, group_attribute, mb_1)

    """N1数据预处理"""
    # 归一化
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    N1['value_normalized'] = scaler_1.fit_transform(N1[[aggregation_attribute]])

    """定义small group merge需要的参数"""
    # 缩放因子
    scaling_factor = 1
    # 相似性阈值
    similar_threshold = 0.5

    small_group_list = []
    merge_info = {}
    final_groups = set()

    """加噪后CV"""
    simple_sizes_list = []
    estimated_frequency_list = []

    data = list(N1[[group_attribute, 'value_normalized']].itertuples(index=False, name=None))

    attempts = 0
    while len(simple_sizes_list) < 1 and attempts < 1000:
        try:
            # 扰动数据
            estimated_frequencies, estimated_means, estimated_variances = LDPGroupBy_samg.group_perturb_phase1(
                data, key_space, epsilon)

            # 检查方差估计中是否存在负值，由于浮点数精度问题，所以会存在负值方差
            if any(variance < 0 for variance in estimated_variances.values()):
                attempts += 1
                continue

            # 使用逆归一化公式还原均值，均值方差和每组数值的方差到原始数据大小
            restored_means = {key: scaler_1.inverse_transform([[m_k]]).item() for key, m_k
                              in estimated_means.items()}
            restored_variances = {key: variance * ((value_max - value_min) ** 2) for key, variance in
                                  estimated_variances.items()}

            """进行small group 合并处理"""
            updated_frequencies, updated_means, updated_variances, final_groups, merge_info = similarity_based_group_merging(estimated_frequencies, estimated_means, estimated_variances, scaling_factor, similar_threshold)

            print("最终的组集合:", final_groups)

            for merged_group, original_groups in merge_info.items():
                print(f"合并组 {merged_group} 由以下组合并而成: {original_groups}")
                
            # 迭代计算各组的采样量（不需要使用逆归一化均值和方差）
            simple_size = iterative_sample_allocation(updated_means, updated_variances, mb_2, final_groups, epsilon)

            if all(value > 0 for value in simple_size.values()):
                simple_sizes_list.append(simple_size)
                estimated_frequency_list.append(updated_frequencies)
            else:
                print(f"第{len(simple_sizes_list) + 1}次扰动结果无效，各分组样本量包含负值，跳过")

            attempts += 1
        except Exception as e:
            print(f"第{len(simple_sizes_list) + 1}次加噪后各分组样本量计算失败: {e}")
            attempts += 1

    average_simple_sizes = {}
    for key in simple_sizes_list[0].keys():
        average_simple_sizes[key] = np.mean([gamma_values[key] for gamma_values in simple_sizes_list])

    average_estimated_frequency = {}
    for key in estimated_frequency_list[0].keys():
        average_estimated_frequency[key] = np.mean([estimated_frequencies[key] for estimated_frequencies in estimated_frequency_list])

    """阶段2"""
    """采样样本"""
    if merge_info:
        N2 = update_group_names(N2, merge_info, group_attribute)

    # 使用分层抽样
    sample_2 = LDPGroupBy_samg.stratified_sampling_probability(N2, average_simple_sizes, group_attribute)
    
    """对样本数据添加噪声"""
    # 进行归一化和离散化
    scaler_2 = MinMaxScaler(feature_range=(-1, 1))
    sample_2['value_normalized'] = scaler_2.fit_transform(sample_2[[aggregation_attribute]])
    sample_2['value_discretized'] = sample_2['value_normalized'].apply(discretize_value)
    
    sample_data = list(sample_2[[group_attribute, 'value_discretized']].itertuples(index=False, name=None))
    
    group_stats = table_simple.groupby(group_attribute).agg({
        aggregation_attribute: ['mean', 'count']
    }).reset_index()
    group_stats.columns = [group_attribute, 'mean', 'count']
    total_count = group_stats['count'].sum()
    group_stats['frequency'] = group_stats['count'] / total_count

    real_result = group_stats.set_index(group_attribute)['mean'].to_dict()
    group_frequencies = group_stats.set_index(group_attribute)['frequency'].to_dict()

    if merge_info:
        merged_real_result = {}
        merged_frequencies = {}

        for merged_group, original_groups in merge_info.items():
            total_freq = sum(group_frequencies[g] for g in original_groups)
            merged_mean = sum(real_result[g] * group_frequencies[g] for g in original_groups) / total_freq
            merged_real_result[merged_group] = merged_mean
            merged_frequencies[merged_group] = total_freq

        unmerged_groups = set(real_result.keys()) - set(sum(merge_info.values(), []))
        for group in unmerged_groups:
            merged_real_result[group] = real_result[group]
            merged_frequencies[group] = group_frequencies[group]

        real_result = merged_real_result
        group_frequencies = merged_frequencies
    print(f"各分组真实均值：{real_result}")
    print(f"各分组频率：{group_frequencies}")

    relative_errors_list = []
    sample_means_list = []
    
    for i in range(10):
        try:
            sample_estimated_frequencies, sample_estimated_means = LDPGroupBy_samg.group_perturb_phase2(
                sample_data, list(final_groups), epsilon)

            sample_restored_means = {key: scaler_2.inverse_transform([[m_k]]).item() for key, m_k
                                     in sample_estimated_means.items()}

            sample_means_list.append(sample_restored_means)

            sample_restored_means_series = pd.Series(sample_restored_means)
            real_result_series = pd.Series(real_result)
            relative_errors = (np.abs(sample_restored_means_series - real_result_series) / real_result_series).to_dict()
            relative_errors_list.append(relative_errors)

        except Exception as e:
            print(f"扰动样本数据失败: {e}")
    
    average_restored_means = pd.DataFrame(sample_means_list).mean().to_dict()
    print("样本估计均值：", average_restored_means)
    
    average_relative_errors = pd.DataFrame(relative_errors_list).mean().to_dict()
    print("相对误差：", average_relative_errors)
    
    average_all_relative_errors = np.mean(list(average_relative_errors.values()))
    print("所有组的平均相对误差:", average_all_relative_errors)

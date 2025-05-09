from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from experiment.mamg import LDPGroupBy_mamg


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

    # 初始化各组均匀分配
    s_i_dict = {key: (total_budget / d) for key in key_space}

    # 迭代更新各组的采样量
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

        new_s_i_dict = {}
        for key in key_space:
            new_s_i_dict[key] = total_budget * (partial_dict[key] / sum_partial)

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
    计算 small group 之间的均值和方差的欧式距离
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

    S = LDPGroupBy_mamg.identify_small_groups(groups_frequencies, scaling_factor)

    # 初始化: G' ← G \ S
    G_prime = set(groups_frequencies.keys()) - set(S)

    # 只有0个或1个小组
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
        new_frequency, new_mean, new_variance = LDPGroupBy_mamg.calculate_new_mean_and_variance(
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
    # 读取 CSV 文件
    file_path = "../../dataset/retirement/retirement_expanded.csv"
    table = pd.read_csv(file_path)

    # 定义分组查询所需参数
    group_by_attributes = ['group_num', 'group_sex']  # 替换为实际的分组列名
    aggregation_attribute_1 = 'agg_value_1'  # 替换为实际的聚合列名
    aggregation_attribute_2 = 'agg_value_2'  # 替换为实际的聚合列名
    memory_budget = 636740  # 总样本容量= 10% * 用户总数
    alpha = 0.4  # phase1 和 phase2 分配的样本数比例

    # 计算phase1和phase2的样本数
    mb_1_1 = int(memory_budget * alpha / 2)
    mb_1_2 = int(memory_budget * alpha / 2)
    mb_2 = int(memory_budget * (1 - alpha))

    # 定义差分隐私参数
    epsilon = 2

    table_simple = table[[*group_by_attributes, aggregation_attribute_1, aggregation_attribute_2]].copy()
    table_simple = table_simple.dropna()
    group_attribute = 'group_key'
    table_simple[group_attribute] = table_simple[group_by_attributes[0]].astype(str).str.strip()
    for attr in group_by_attributes[1:]:
        table_simple[group_attribute] += '-' + table_simple[attr].astype(str).str.strip()
    unique_group_key = table_simple[group_attribute].unique().tolist()

    total_users = len(table_simple)

    value_max_1 = table_simple[aggregation_attribute_1].max()
    value_min_1 = table_simple[aggregation_attribute_1].min()
    value_max_2 = table_simple[aggregation_attribute_2].max()
    value_min_2 = table_simple[aggregation_attribute_2].min()

    """在N1中均匀采样mb_1的数据量"""
    N1_agg1, N1_agg2, N2, N1_sample_size_1, N1_sample_size_2 = LDPGroupBy_mamg.uniform_sampling(table_simple,
                                                                                                group_attribute,
                                                                                                aggregation_attribute_1,
                                                                                                aggregation_attribute_2,
                                                                                                mb_1_1, mb_1_2)

    """N1数据预处理"""
    N1_agg2[group_attribute] = N1_agg2[group_attribute].apply(lambda x: str(x) + '+2')

    # 归一化
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    scaler_2 = MinMaxScaler(feature_range=(0, 1))
    N1_agg1['value_normalized_1'] = scaler_1.fit_transform(N1_agg1[[aggregation_attribute_1]])
    N1_agg2['value_normalized_2'] = scaler_2.fit_transform(N1_agg2[[aggregation_attribute_2]])

    """定义small group merge需要的参数"""
    # 缩放因子
    scaling_factor = 1
    # 相似性阈值
    similar_threshold = 0.5

    small_group_list_1 = []
    small_group_list_2 = []
    merge_info_1 = {}
    merge_info_2 = {}
    final_groups_1 = set()
    final_groups_2 = set()

    """加噪后CV"""
    key_space_1 = table_simple[group_attribute].unique().tolist()
    n_key_1 = len(key_space_1)
    key_space_2 = N1_agg2[group_attribute].unique().tolist()
    n_key_2 = len(key_space_2)
    key_space = key_space_1 + key_space_2
    n_key = len(key_space)

    simple_sizes_list = []
    estimated_frequency_list = []
    estimated_means_list = []

    data_1 = list(N1_agg1[[group_attribute, 'value_normalized_1']].itertuples(index=False, name=None))
    data_2 = list(N1_agg2[[group_attribute, 'value_normalized_2']].itertuples(index=False, name=None))

    attempts = 0
    while len(simple_sizes_list) < 1 and attempts < 1000:
        try:
            # 扰动数据
            estimated_frequencies_1, estimated_means_1, estimated_variances_1 = LDPGroupBy_mamg.group_perturb_phase1(
                data_1, key_space_1, epsilon)
            estimated_frequencies_2, estimated_means_2, estimated_variances_2 = LDPGroupBy_mamg.group_perturb_phase1(
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

            updated_frequencies_1, updated_means_1, updated_variances_1, final_groups_1, merge_info_1 = similarity_based_group_merging(estimated_frequencies_1, estimated_means_1, estimated_variances_1, scaling_factor, similar_threshold)
            print("agg1最终的组集合:", final_groups_1)

            updated_frequencies_2, updated_means_2, updated_variances_2, final_groups_2, merge_info_2 = similarity_based_group_merging(estimated_frequencies_2, estimated_means_2, estimated_variances_2, scaling_factor, similar_threshold)
            print("agg2最终的组集合:", final_groups_2)

            estimated_means = {}
            for key in final_groups_1:
                estimated_means[key] = updated_means_1[key]
            for key in final_groups_2:
                estimated_means[key] = updated_means_2[key]

            estimated_variances = {}
            for key in final_groups_1:
                estimated_variances[key] = updated_variances_1[key]
            for key in final_groups_2:
                estimated_variances[key] = updated_variances_2[key]

            estimated_frequencies = {}
            for key in final_groups_1:
                estimated_frequencies[key] = updated_frequencies_1[key]
            for key in final_groups_2:
                estimated_frequencies[key] = updated_frequencies_2[key]

            final_groups = final_groups_1.union(final_groups_2)

            # 迭代计算各组的采样量（不需要使用逆归一化均值和方差）
            simple_size = iterative_sample_allocation(estimated_means, estimated_variances, mb_2, final_groups, len(final_groups_1), epsilon)

            if all(value > 0 for value in simple_size.values()):
                simple_sizes_list.append(simple_size)
                estimated_frequency_list.append(estimated_frequencies)
                estimated_means_list.append(estimated_means)
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
        
    average_estimated_means = {}
    for key in estimated_means_list[0].keys():
        average_estimated_means[key] = np.mean([estimated_means[key] for estimated_means in estimated_means_list])

    """阶段2"""
    """采样样本"""
    simple_size_agg1 = {}
    simple_size_agg2 = {}
    for key in final_groups_1:
        simple_size_agg1[key] = average_simple_sizes[key]

    simple_size_agg2 = {}
    for key in final_groups_2:
        original_key = key.replace('+2', '')
        simple_size_agg2[original_key] = average_simple_sizes[key]

    merge_info_2_original = {}
    for key, values in merge_info_2.items():
        original_key = key.replace('+2', '')
        original_values = [value.replace('+2', '') for value in values]
        merge_info_2_original[original_key] = original_values
    merge_info_2 = merge_info_2_original

    final_groups_2_original = set()
    for key in final_groups_2:
        original_key = key.replace('+2', '')
        final_groups_2_original.add(original_key)
    final_groups_2 = final_groups_2_original

    average_estimated_means_agg1 = {}
    average_estimated_means_agg2 = {}

    for key in key_space_1:
        if key in average_estimated_means:
            average_estimated_means_agg1[key] = average_estimated_means[key]

    for key in average_estimated_means:
        if '+2' in key:
            original_key = key.replace('+2', '')
            average_estimated_means_agg2[original_key] = average_estimated_means[key]

    N2_shuffled = N2.sample(frac=1, random_state=42)  # 随机打乱N2数据
    half_size = len(N2_shuffled) // 2
    N2_agg1 = N2_shuffled.iloc[:half_size].copy()
    N2_agg2 = N2_shuffled.iloc[half_size:].copy()

    if merge_info_1:
        N2_agg1 = update_group_names(N2_agg1, merge_info_1, group_attribute)

    if merge_info_2:
        N2_agg2 = update_group_names(N2_agg2, merge_info_2, group_attribute)

    # 使用分层抽样
    sample_2_agg1 = LDPGroupBy_mamg.stratified_sampling_probability_optimized(
        N2_agg1,
        simple_size_agg1,
        group_attribute
    )
    sample_2_agg2 = LDPGroupBy_mamg.stratified_sampling_probability_optimized(
        N2_agg2,
        simple_size_agg2,
        group_attribute
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
        sample_2_agg1[[group_attribute, 'value_discretized_1']].itertuples(index=False, name=None))
    sample_data_agg2 = list(
        sample_2_agg2[[group_attribute, 'value_discretized_2']].itertuples(index=False, name=None))

    group_stats_agg1 = table_simple.groupby(group_attribute).agg({
        aggregation_attribute_1: ['mean', 'count']
    }).reset_index()
    group_stats_agg1.columns = [group_attribute, 'mean', 'count']
    total_count_agg1 = group_stats_agg1['count'].sum()
    group_stats_agg1['frequency'] = group_stats_agg1['count'] / total_count_agg1

    real_result_agg1 = group_stats_agg1.set_index(group_attribute)['mean'].to_dict()
    group_frequencies_agg1 = group_stats_agg1.set_index(group_attribute)['frequency'].to_dict()

    if merge_info_1:
        merged_real_result_agg1 = {}
        merged_frequencies_agg1 = {}

        for merged_group, original_groups in merge_info_1.items():
            total_freq = sum(group_frequencies_agg1[g] for g in original_groups if g in group_frequencies_agg1)
            if total_freq > 0:
                merged_mean = sum(real_result_agg1[g] * group_frequencies_agg1[g]
                                  for g in original_groups if g in group_frequencies_agg1) / total_freq
                merged_real_result_agg1[merged_group] = merged_mean
                merged_frequencies_agg1[merged_group] = total_freq
        
        all_original_groups = set(sum(merge_info_1.values(), []))
        unmerged_groups = set(real_result_agg1.keys()) - all_original_groups
        for group in unmerged_groups:
            merged_real_result_agg1[group] = real_result_agg1[group]
            merged_frequencies_agg1[group] = group_frequencies_agg1[group]
            
        real_result_agg1 = merged_real_result_agg1
        group_frequencies_agg1 = merged_frequencies_agg1

    group_stats_agg2 = table_simple.groupby(group_attribute).agg({
        aggregation_attribute_2: ['mean', 'count']
    }).reset_index()
    group_stats_agg2.columns = [group_attribute, 'mean', 'count']
    total_count_agg2 = group_stats_agg2['count'].sum()
    group_stats_agg2['frequency'] = group_stats_agg2['count'] / total_count_agg2

    real_result_agg2 = group_stats_agg2.set_index(group_attribute)['mean'].to_dict()
    group_frequencies_agg2 = group_stats_agg2.set_index(group_attribute)['frequency'].to_dict()

    if merge_info_2:
        merged_real_result_agg2 = {}
        merged_frequencies_agg2 = {}
        
        for merged_group, original_groups in merge_info_2.items():
            total_freq = sum(group_frequencies_agg2[g] for g in original_groups if g in group_frequencies_agg2)
            if total_freq > 0:
                merged_mean = sum(real_result_agg2[g] * group_frequencies_agg2[g]
                                  for g in original_groups if g in group_frequencies_agg2) / total_freq
                merged_real_result_agg2[merged_group] = merged_mean
                merged_frequencies_agg2[merged_group] = total_freq
        
        all_original_groups = set(sum(merge_info_2.values(), []))
        unmerged_groups = set(real_result_agg2.keys()) - all_original_groups
        for group in unmerged_groups:
            merged_real_result_agg2[group] = real_result_agg2[group]
            merged_frequencies_agg2[group] = group_frequencies_agg2[group]
            
        real_result_agg2 = merged_real_result_agg2
        group_frequencies_agg2 = merged_frequencies_agg2

    print(f"各分组真实均值：agg1={real_result_agg1}, agg2={real_result_agg2}")

    relative_errors_list_agg1 = []
    relative_errors_list_agg2 = []
    sample_means_list_agg1 = []
    sample_means_list_agg2 = []

    for i in range(10):
        try:
            sample_estimated_frequencies_agg1, sample_estimated_means_agg1 = LDPGroupBy_mamg.group_perturb_phase2(
                sample_data_agg1, list(final_groups_1), epsilon)
            sample_estimated_frequencies_agg2, sample_estimated_means_agg2 = LDPGroupBy_mamg.group_perturb_phase2(
                sample_data_agg2, list(final_groups_2), epsilon)

            # 使用逆归一化公式还原均值和方差到原始数据大小
            sample_restored_means_agg1 = {key: scaler_3.inverse_transform([[m_k]]).item() for key, m_k
                                          in sample_estimated_means_agg1.items()}
            sample_restored_means_agg2 = {key: scaler_4.inverse_transform([[m_k]]).item() for key, m_k
                                          in sample_estimated_means_agg2.items()}

            sample_means_list_agg1.append(sample_restored_means_agg1)
            sample_means_list_agg2.append(sample_restored_means_agg2)

            sample_restored_means_series_agg1 = pd.Series(sample_restored_means_agg1)
            real_result_series_agg1 = pd.Series(real_result_agg1)
            relative_errors_agg1 = (np.abs(sample_restored_means_series_agg1 - real_result_series_agg1) / real_result_series_agg1).to_dict()

            sample_restored_means_series_agg2 = pd.Series(sample_restored_means_agg2)
            real_result_series_agg2 = pd.Series(real_result_agg2)
            relative_errors_agg2 = (np.abs(sample_restored_means_series_agg2 - real_result_series_agg2) / real_result_series_agg2).to_dict()

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

    avg_error_agg1 = np.mean(list(average_relative_errors_agg1.values()))
    avg_error_agg2 = np.mean(list(average_relative_errors_agg2.values()))
    print(f"第一个聚合属性的总体平均相对误差: {avg_error_agg1}")
    print(f"第二个聚合属性的总体平均相对误差: {avg_error_agg2}")

    average_all_relative_errors = (avg_error_agg1 + avg_error_agg2) / 2
    print(f"所有组的总体平均相对误差: {average_all_relative_errors}")


import numpy as np
import pandas as pd
from typing import List, Tuple


def general_random_response(epsilon, key, key_space):
    """对键进行GRR扰动"""
    d = len(key_space)
    a_prob = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    b_prob = 1 / (np.exp(epsilon) + d - 1)
    item_index = {item: index for index, item in enumerate(key_space)}

    if key not in key_space:
        raise Exception(f"ERR: the input key={key} is not in the key_space={key_space}.")

    probability_arr = np.full(shape=d, fill_value=b_prob)
    probability_arr[item_index[key]] = a_prob

    return np.random.choice(a=key_space, p=probability_arr)


def value_perturbation_sw(value, p, q, b):
    """
    对值进行 SW 机制扰动
    """

    p_area = p * 2 * b  # [v-b,v+b]区间的概率

    random_prob = np.random.random()

    if random_prob <= p_area:
        # 第一种情况：扰动到[v-b,v+b]
        return np.random.uniform(value - b, value + b)
    else:
        # 第二种情况：以概率 q * 1 扰动到[-b,v-b]U[v+b,1+b]
        r2 = np.random.random()
        if r2 < value:
            return np.random.uniform(-b, value - b)
        else:
            return np.random.uniform(value + b, 1 + b)


def value_perturbation_rr(value, p):
    """对值进行随机响应扰动"""
    if np.random.binomial(1, p) == 1:
        return value
    else:
        return -value


def grr_sw(key_value_pair, key_space, epsilon_1, p, q, b):
    """
    GRR-SW 机制
    键的扰动采用GRR，以p的概率保持不变时，value使用SW机制；
    以q的概率变为其他时，value的fake value从均匀分布 U(-b,1+b)中抽取
    """
    key, value = key_value_pair
    perturbed_key = general_random_response(epsilon_1, key, key_space)
    perturbed_value = value_perturbation_sw(value, p, q, b) if perturbed_key == key else np.random.uniform(-b, 1 + b)

    return perturbed_key, perturbed_value


def grr_rr(key_value_pair, key_space, epsilon_1, p):
    """PCKV-GRR 机制"""
    key, value = key_value_pair
    perturbed_key = general_random_response(epsilon_1, key, key_space)
    perturbed_value = value_perturbation_rr(value, p) if perturbed_key == key else np.random.choice([1, -1], p=[0.5, 0.5])
    return perturbed_key, perturbed_value


def group_perturb_phase1(data, key_space, epsilon):
    """
    对数据集中的每个<g,v>对进行 GRR-SW 扰动,然后计算每个组的均值和方差
    """
    n_user = len(data)
    n_key = len(key_space)
    epsilon_1 = epsilon / 2
    epsilon_2 = epsilon / 2
    a = np.exp(epsilon_1) / (np.exp(epsilon_1) + n_key - 1)
    c = 1 / (np.exp(epsilon_1) + n_key - 1)
    # b 表示扰动的波长
    b = (epsilon_2 * np.exp(epsilon_2) - np.exp(epsilon_2) + 1) / (
            2 * np.exp(epsilon_2) * (np.exp(epsilon_2) - 1 - epsilon_2))
    w = b * 2
    p = np.exp(epsilon_2) / (2 * b * np.exp(epsilon_2) + 1)
    q = 1 / (2 * b * np.exp(epsilon_2) + 1)

    # 统计原始数据中每个key对应的value数量
    group_count = {key: 0 for key in key_space}
    for kv_pair in data:
        key = kv_pair[0]
        group_count[key] += 1

    # 对键值进行扰动
    perturbed_data = [grr_sw(kv_pair, key_space, epsilon_1, p, q, b) for kv_pair in data]

    """
    服务器端进行key频率估计和value的方差估计
    """
    # 对key进行频数统计
    noisy_counts = {key: 0 for key in key_space}
    noisy_values = {key: [] for key in key_space}
    for key, value in perturbed_data:
        noisy_counts[key] += 1
        noisy_values[key].append(value)

    # 定义估计频率，均值，方差
    noisy_key_count = {key: 0 for key in key_space}
    estimated_frequencies = {key: 0 for key in key_space}
    estimated_means = {key: 0 for key in key_space}
    estimated_variances = {key: 0 for key in key_space}

    for key in key_space:
        if noisy_counts[key] > 0:
            key_count = noisy_counts[key]
            frequency = (key_count / n_user - c) / (a - c)
            noisy_key_count[key] = frequency
            # 矫正频率估计
            estimated_frequency = min(max(frequency, 1 / n_user), 1)
            estimated_frequencies[key] = estimated_frequency

            # 获取观测均值
            observed_mean = np.mean(noisy_values[key])
            observed_freq = key_count / n_user

            # 计算估计均值
            numerator = (observed_mean * observed_freq) - (c * (1 - frequency) / 2) - (a * frequency * q * (b + 1/2))
            denominator = a * frequency * 2 * b * (p - q)
            estimated_mean = numerator / denominator
            estimated_means[key] = estimated_mean

            # 计算估计方差
            observed_variance = np.var(noisy_values[key])
            term_1 = observed_freq * (observed_variance + observed_mean ** 2)
            term_2 = c * (1 - frequency) * ((1 + b + b ** 2) / 3)
            term_3 = a * frequency * ((b ** 2 + q * (2 * b + 1) * (b + 1)) / 3)
            term_4 = estimated_mean ** 2
            estimated_variance = ((term_1 - term_2 - term_3) / denominator) - term_4
            estimated_variances[key] = estimated_variance

    return estimated_frequencies, estimated_means, estimated_variances


def group_perturb_phase2(data, key_space, epsilon):
    """
    对数据集中的每个<g,v>对进行 PCKV-GRR 扰动,然后计算每个组的均值
    """
    n_user = len(data)
    n_key = len(key_space)
    epsilon_1 = np.log((np.exp(epsilon) + 1) / 2)
    epsilon_2 = epsilon
    a = np.exp(epsilon_1) / (np.exp(epsilon_1) + n_key - 1)
    c = 1 / (np.exp(epsilon_1) + n_key - 1)
    p = np.exp(epsilon_2) / (np.exp(epsilon_2) + 1)

    # 对键值进行扰动
    perturbed_data = [grr_rr(kv_pair, key_space, epsilon_1, p) for kv_pair in data]

    """服务器端"""
    # 聚合器端进行均值估计
    key_counts = {key: 0 for key in key_space}
    positive_counts = {key: 0 for key in key_space}
    negative_counts = {key: 0 for key in key_space}

    for key, value in perturbed_data:
        key_counts[key] += 1
        if value == 1:
            positive_counts[key] += 1
        else:
            negative_counts[key] += 1

    # 计算估计均值
    estimated_frequencies = {}
    estimated_means = {}

    for key in key_space:
        if key_counts[key] > 0:
            n1 = positive_counts[key]
            n2 = negative_counts[key]
            frequency = ((n1 + n2) / n_user - c) / (a - c)
            # 矫正频率估计
            estimated_frequency = min(max(frequency, 1 / n_user), 1)
            estimated_frequencies[key] = estimated_frequency

            # 计算估计均值
            N = n_user * estimated_frequency
            # 生成矩阵A 和 向量 b_n
            A = np.array([
                [a * p - c / 2, a * (1 - p) - c / 2],
                [a * (1 - p) - c / 2, a * p - c / 2]
            ])
            b_n = np.array([n1, n2]) - n_user * c / 2

            # 求解线性方程 A * est = b_n 得到 est (均值估计的向量)
            est = np.linalg.solve(A, b_n)

            # 限制 n1 和 n2 的估计值
            n1_est = max(min(float(est[0]), N), 1)
            n2_est = max(min(float(est[1]), N), 1)

            # 计算 m_k (均值)
            m_k = (n1_est - n2_est) / N

            estimated_means[key] = m_k

    return estimated_frequencies, estimated_means


def stratified_sampling_probability(data, sample_size_dict, group_by_attributes):
    """
    基于概率的分层采样
    """
    # 计算各组的采样概率： p_g = s_g / n_g
    prob_dict = {}
    for group, sample_size in sample_size_dict.items():
        n = data[group_by_attributes].value_counts().to_dict()
        prob_dict[group] = (sample_size / n) if n > 0 else 0.0

    # 遍历所有数据，进行采样
    sampled_data = []
    for _, row in data.iterrows():
        group = row[group_by_attributes]
        if np.random.random() < prob_dict[group]:
            sampled_data.append(row)

    return pd.DataFrame(sampled_data, columns=data.columns)


def uniform_sampling(data, group_column, sample_size, random_state: int = 42):
    """
    根据各组的概率进行均匀采样
    """
    if group_column not in data.columns:
        raise ValueError(f"列 '{group_column}' 在 DataFrame 中不存在")

    group_frequencies = data[group_column].value_counts(normalize=True)

    group_sample_counts = (group_frequencies * sample_size).astype(int)

    remaining_samples = sample_size - group_sample_counts.sum()
    if remaining_samples > 0:
        # 将剩余的样本分配给频率最高的组
        top_groups = group_frequencies.nlargest(remaining_samples).index
        for group in top_groups:
            group_sample_counts[group] += 1

    sampled_list = []
    remaining_list = []
    for group, count in group_sample_counts.items():
        group_df = data[data[group_column] == group]
        if len(group_df) < count:
            raise ValueError(
                f"分组 '{group}' 的数据量不足以采样 {count} 条。"
            )
        sampled = group_df.sample(n=count, random_state=random_state)
        sampled_list.append(sampled)
        remaining = group_df.loc[~group_df.index.isin(sampled.index)]
        remaining_list.append(remaining)

    sampled_data = pd.concat(sampled_list, ignore_index=True)
    remaining_data = pd.concat(remaining_list, ignore_index=True)

    return sampled_data, remaining_data, group_sample_counts


def identify_small_groups(estimated_frequencies: dict, scaling_factor: float = 0.3) -> List[str]:
    """
    使用 estimated_frequencies 识别 small group
    """
    num_groups = len(estimated_frequencies)  # 组数 g
    threshold = scaling_factor / num_groups  # 计算阈值 k

    # 识别 small group
    small_groups = [key for key, freq in estimated_frequencies.items() if freq < threshold]

    return small_groups


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
                
                # 更新距离矩阵
                distance_matrix.at[key1, key2] = distance
                distance_matrix.at[key2, key1] = distance
                
                dependence_list.append((key1, key2, distance))
                d_max = max(d_max, distance)

    dependence_list.sort(key=lambda x: x[2])

    return distance_matrix, dependence_list, d_max


def calculate_new_mean_and_variance(key1, key2, groups_frequencies, groups_means, groups_variances):
    """
    重新计算合并后的组的均值和方差。
    """
    f_i = groups_frequencies[key1]
    f_j = groups_frequencies[key2]
    
    # 新组频率
    f_new = f_i + f_j
    
    # 新组均值
    mu_i = groups_means[key1]
    mu_j = groups_means[key2]
    mu_new = (f_i * mu_i + f_j * mu_j) / f_new
    
    # 新组方差
    var_i = groups_variances[key1]
    var_j = groups_variances[key2]
    
    # 计算新组的方差
    var_new = (f_i * (var_i + mu_i**2) + f_j * (var_j + mu_j**2)) / f_new - mu_new**2
    
    return f_new, mu_new, var_new

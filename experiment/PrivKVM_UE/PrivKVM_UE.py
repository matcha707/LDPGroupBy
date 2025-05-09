import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def random_response(bit_array: np.ndarray, prob, q=None):
    """
    对数组中每个值执行随机响应机制
    """
    q = 1 - prob if q is None else q
    if isinstance(bit_array, int):
        probability = prob if bit_array == 1 else q
        return np.random.binomial(n=1, p=probability)
    # 对数组中的每个元素应用随机响应
    return np.where(bit_array == 1, np.random.binomial(1, prob, len(bit_array)),
                    np.random.binomial(1, q, len(bit_array)))


def vp_UE(bit_array: np.ndarray, epsilon):
    """
    一元编码函数
    """
    if not isinstance(bit_array, np.ndarray):
        raise Exception("Type Error: ", type(bit_array))

    p_ue = 1 / 2
    q_ue = 1 / (np.exp(epsilon) + 1)
    return random_response(bit_array, p_ue, q_ue)


def vp_GRR(v, epsilon, v_space):
    """对值进行广义随机响应扰动"""
    d = len(v_space)
    a_prob = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    b_prob = 1 / (np.exp(epsilon) + d - 1)
    item_index = {item: index for index, item in enumerate(v_space)}

    if v not in v_space:
        raise Exception(f"ERR: the input key={v} is not in the key_space={v_space}.")

    probability_arr = np.full(shape=d, fill_value=b_prob)
    probability_arr[item_index[v]] = a_prob

    return np.random.choice(a=v_space, p=probability_arr)


def GVPP(k_v, epsilon_1, epsilon_2, is_than, v_space):
    """
    GVPP扰动协议：键key使用RR，值value使用GRR或UE
    """
    k = k_v[0]
    v = k_v[1]
    kp = 0
    vp = 0
    boundary = np.exp(epsilon_1) / (1 + np.exp(epsilon_1))

    if np.random.binomial(n=1, p=boundary):
        kp = 1
        if is_than:
            vp = vp_GRR(v, epsilon_2, v_space)
        else:
            vp = vp_UE(v, epsilon_2)
    else:
        kp = 0
        vp = 0

    return kp, vp


def get_interval_boundaries(left, right, g):
    """
    将值域[-1, 1]均匀划分成g-1个区间，并返回区间的分界点
    """
    bin_width = (right - left) / (g - 1)

    # 计算区间的分界点
    bin_point = np.linspace(left, right, g)
    bin_point = np.round(bin_point, 1)

    return bin_point


def get_value_space(boundaries):
    """
    根据 boundaries 和公式构建 value_space
    """
    # 获取区间的个数
    g = len(boundaries)

    value_space = []

    # 添加 x_1^+，即最小的离散化值
    value_space.append(f'{boundaries[0]}^+')

    # 对于每个内部分界点 x_2, x_3, ..., x_(g-1)，添加 x_i^- 和 x_i^+
    for i in range(1, g - 1):
        value_space.append(f'{boundaries[i]}^-')  # 添加 x_i^-
        value_space.append(f'{boundaries[i]}^+')  # 添加 x_i^+

    # 添加 x_g^-，即最大的离散化值
    value_space.append(f'{boundaries[g - 1]}^-')

    return value_space


def discretize_value(v, boundaries):
    """
    将值 v 离散化为相应的区间边界值，并在离散化后进行标记
    """
    if v <= boundaries[0]:
        return boundaries[0], f'{boundaries[0]}^+'
    elif v >= boundaries[-1]:
        return boundaries[-1], f'{boundaries[-1]}^-'

    # 找到包含 v 的区间 [x_i, x_{i+1}]
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= v <= boundaries[i + 1]:
            prob = (boundaries[i + 1] - v) / (boundaries[i + 1] - boundaries[i])
            if np.random.rand() < prob:
                # 映射到左边界
                return boundaries[i], f'{boundaries[i]}^+'
            else:
                # 映射到右边界
                return boundaries[i + 1], f'{boundaries[i + 1]}^-'


def encode_discretized_flag(iscretized_flag, value_space):
    """
    根据 discretized_flag 在 value_space 中生成编码向量
    """
    encoding_vector = np.zeros(len(value_space), dtype=int)
    # 找到 discretized_flag 在 value_space 中的位置，并将该位置置为 1
    if iscretized_flag in value_space:
        index = value_space.index(iscretized_flag)
        encoding_vector[index] = 1
    return encoding_vector


def privkvm_perturb(group_data, total_user, epsilon_1, epsilon_2):
    p = np.exp(epsilon_1) / (1 + np.exp(epsilon_1))
    q = 1 / (1 + np.exp(epsilon_1))

    """对每个组进行扰动和统计"""
    estimated_frequency = {}  # 估计频率
    estimated_mean = {}  # 估计均值
    for group, values in group_data.items():
        all_kv = [(1, v) for v in values]
        all_kvp = [GVPP(kv, epsilon1, epsilon2, is_than, value_space) for kv in all_kv]

        have = sum(1 for kv in all_kvp if kv[0] == 1)

        """计算并校准每个key的频率"""
        f = have / len(all_kv)
        f = (p - 1 + f) / (2 * p - 1)
        fre = f * (len(all_kv) / total_user)
        estimated_frequency[group] = fre

        n_k = have

        """均值估计校准"""
        if is_than:
            # value使用GRR方式扰动
            perturbed_counts = {}
            for v in [kv[1] for kv in all_kvp if kv[0] == 1]:
                perturbed_counts[v] = perturbed_counts.get(v, 0) + 1

            calibrated_counts = {}  # 存储校准后的计数
            for v in value_space:
                # 校准
                c_tilde = perturbed_counts.get(v, 0)
                c = ((L - 1 + np.exp(epsilon_2)) * c_tilde - n_k) / (np.exp(epsilon_2) - 1)
                calibrated_counts[v] = max(0, c)
        else:
            # value使用UE方式扰动
            perturbed_counts = np.zeros(L)
            for encoded_vec in [kv[1] for kv in all_kvp if kv[0] == 1]:
                perturbed_counts += encoded_vec

            calibrated_counts = {}
            for i in range(L):
                c_tilde = perturbed_counts[i]
                # 校准
                c = (2 * (np.exp(epsilon_2) + 1) * c_tilde - 2 * n_k) / (np.exp(epsilon_2) - 1)
                if i % 2 == 0:  # 偶数索引对应 x_i^+
                    boundary_idx = i // 2
                    calibrated_counts[f'{boundaries[boundary_idx]}^+'] = max(0, c)
                else:
                    boundary_idx = i // 2 + 1
                    calibrated_counts[f'{boundaries[boundary_idx]}^-'] = max(0, c)

        bucket_counts = []
        bucket_means = []

        for i in range(g - 1):
            # 获取区间的两个边界点对应的校准计数
            c_i_plus = calibrated_counts.get(f'{boundaries[i]}^+', 0)
            c_i_plus_1_minus = calibrated_counts.get(f'{boundaries[i + 1]}^-', 0)

            # 计算每个区间计数 Φ_k^i
            phi_k = ((c_i_plus + c_i_plus_1_minus) * total_user / n_k -
                     total_user * (1 - fre) * (1 - p) / (g - 1)) / p
            bucket_counts.append(max(0, phi_k))

            # 计算每个区间平均值 Ψ_k^i
            if (c_i_plus + c_i_plus_1_minus) > 0:
                psi_k = (boundaries[i] * c_i_plus + boundaries[i + 1] * c_i_plus_1_minus) / (
                        c_i_plus + c_i_plus_1_minus)
                bucket_means.append(psi_k)
            else:
                bucket_means.append(0)

        # 计算总体平均值
        total_count = sum(bucket_counts)
        m = 0
        if total_count > 0:
            m = sum(count * mean for count, mean in zip(bucket_counts, bucket_means)) / total_count

        estimated_mean[group] = m

    return estimated_frequency, estimated_mean


def stratified_sampling(data: pd.DataFrame, group_column, sample_size: int,
                        random_state: int = 42):
    """
    均匀采样，根据每个组的频率分配样本量
    """
    if group_column not in data.columns:
        raise ValueError(f"列 '{group_column}' 在 DataFrame 中不存在")

    group_frequencies = data[group_column].value_counts(normalize=True)

    group_sample_counts = (group_frequencies * sample_size).astype(int)

    remaining_samples = sample_size - group_sample_counts.sum()
    if remaining_samples > 0:
        top_groups = group_frequencies.nlargest(remaining_samples).index
        for group in top_groups:
            group_sample_counts[group] += 1

    sampled_list = []
    for group, count in group_sample_counts.items():
        group_df = data[data[group_column] == group]
        if len(group_df) < count:
            raise ValueError(
                f"分组 '{group}' 的数据量不足以采样 {count} 条。"
            )
        sampled_list.append(
            group_df.sample(n=count, random_state=random_state)
        )

    sampled_data = pd.concat(sampled_list, ignore_index=True)
    return sampled_data, group_sample_counts


if __name__ == '__main__':
    # 读取 CSV 文件
    file_path = "../../dataset/retirement/retirement_expanded.csv"
    df = pd.read_csv(file_path)

    # 定义分组参数
    group_by_attributes = ['Organization Group Code']  # 替换为实际的分组列名
    aggregation_attribute = 'Retirement'  # 替换为实际的聚合列名

    key_space = df[group_by_attributes[0]].unique().tolist()

    # 定义隐私预算参数
    epsilon1 = 1
    epsilon2 = 1

    """数据预处理"""
    memory_budget = 63674  # 用实际样本容量替换 1%
    table, group_sample_size = stratified_sampling(df, group_by_attributes[0], memory_budget)
    table = table[[group_by_attributes[0], aggregation_attribute]].copy()

    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    table['value_normalized'] = scaler.fit_transform(table[[aggregation_attribute]])

    total_users = len(table)

    # 定义值域划分区间分界点数 g
    g = 6
    L = 2 * (g - 1)  # L表示所有可能离散点数

    # 将值域[-1,1]划分成 g-1 个区间
    left = -1
    right = 1
    # 分界点
    boundaries = get_interval_boundaries(left, right, g)
    # 离散点集合
    value_space = get_value_space(boundaries)

    # 离散化
    table['discretized_value'], table['discretized_flag'] = zip(
        *table['value_normalized'].apply(lambda x: discretize_value(x, boundaries)))

    table['encoded_value'] = table['discretized_flag'].apply(lambda flag: encode_discretized_flag(flag, value_space))

    if epsilon2 >= math.log(L / 2):
        is_than = True
        grouped_data = table.groupby(group_by_attributes)['discretized_flag'].apply(list).to_dict()
    else:
        is_than = False
        grouped_data = table.groupby(group_by_attributes)['encoded_value'].apply(list).to_dict()

    real_result = df.groupby(group_by_attributes)[aggregation_attribute].agg(['mean', 'var'])
    print("各组的真实均值和方差:", real_result)

    n_iterations = 10
    all_means = {key: [] for key in key_space}
    all_relative_errors = {key: [] for key in key_space}
    completed_iterations = 0

    while completed_iterations < n_iterations:
        # 对table进行PrivKVM扰动
        estimated_frequencies, estimated_means = privkvm_perturb(grouped_data, total_users, epsilon1, epsilon2)

        # 使用逆归一化还原均值
        restored_means = {key: scaler.inverse_transform(np.array(m_k).reshape(-1, 1)).item() for key, m_k
                          in estimated_means.items()}

        valid_iteration = True
        for key, mean in restored_means.items():
            if mean < 0:
                valid_iteration = False
                break

        if valid_iteration:
            for key, mean in restored_means.items():
                all_means[key].append(mean)
                true_mean = real_result.loc[key, 'mean']
                relative_error = abs(mean - true_mean) / true_mean
                all_relative_errors[key].append(relative_error)
            completed_iterations += 1

    average_means = {key: np.mean(means) for key, means in all_means.items()}
    print("估计均值:", average_means)

    average_relative_errors = {key: np.mean(errors) for key, errors in all_relative_errors.items()}
    print("相对误差:", average_relative_errors)

    average_all_relative_errors = np.mean(list(average_relative_errors.values()))
    print("所有组的平均相对误差:", average_all_relative_errors)


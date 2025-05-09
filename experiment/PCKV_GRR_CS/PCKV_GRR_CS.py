import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# 对value_normalized进行离散化
def discretize_value(v):
    v = np.clip(v, -1, 1)
    prob = (1 + v) / 2
    return np.random.choice([1, -1], p=[prob, 1 - prob])


def random_response(epsilon, key, key_space):
    """对键进行广义随机响应扰动"""
    d = len(key_space)
    a_prob = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    b_prob = 1 / (np.exp(epsilon) + d - 1)
    item_index = {item: index for index, item in enumerate(key_space)}

    if key not in key_space:
        raise Exception(f"ERR: the input key={key} is not in the key_space={key_space}.")

    probability_arr = np.full(shape=d, fill_value=b_prob)
    probability_arr[item_index[key]] = a_prob

    return np.random.choice(a=key_space, p=probability_arr)


def value_perturbation(value, p):
    """对值进行随机响应扰动"""
    if np.random.binomial(1, p) == 1:
        return value
    else:
        return -value


def pckv_grr(key_value_pair, key_space, epsilon_1, p):
    """PCKV-GRR 机制"""
    key, value = key_value_pair
    perturbed_key = random_response(epsilon_1, key, key_space)
    perturbed_value = value_perturbation(value, p) if perturbed_key == key else np.random.choice([1, -1], p=[0.5, 0.5])

    return perturbed_key, perturbed_value


def group_perturb_phase1(data, key_space, epsilon):
    """
    只对数据集中的key进行扰动,然后计算每个组的频率和频率方差
    """
    n_user = len(data)
    n_key = len(key_space)
    epsilon_1 = epsilon
    a = np.exp(epsilon_1) / (np.exp(epsilon_1) + n_key - 1)
    c = 1 / (np.exp(epsilon_1) + n_key - 1)

    # 统计原始数据中每个key对应的value数量
    group_count = {key: 0 for key in key_space}
    for kv_pair in data:
        key = kv_pair[0]
        group_count[key] += 1

    # 只对key使用GRR进行扰动,value保持不变
    perturbed_data = [(random_response(epsilon_1, kv_pair[0], key_space), kv_pair[1]) for kv_pair in data]

    """
    服务器端
    """
    # 对key进行频数统计
    noisy_counts = {key: 0 for key in key_space}
    noisy_values = {key: [] for key in key_space}
    for key, value in perturbed_data:
        noisy_counts[key] += 1
        noisy_values[key].append(value)

    # 定义频率，频率的方差
    noisy_key_count = {key: 0 for key in key_space}
    estimated_frequencies = {key: 0 for key in key_space}
    frequency_variances = {key: 0 for key in key_space}

    # 进行频率估计，频率的方差估计
    for key in key_space:
        if noisy_counts[key] > 0:
            key_count = noisy_counts[key]
            frequency = (key_count / n_user - c) / (a - c)
            noisy_key_count[key] = frequency
            # 矫正频率估计，防止分母为零
            estimated_frequency = min(max(frequency, 1 / n_user), 1)
            estimated_frequencies[key] = estimated_frequency

            # 频率的方差估计
            sample_variance = (estimated_frequency * (1 - estimated_frequency)) / n_user
            estimated_frequency_variances = (c * (1 - c) / (n_user * (a - c)**2)) + (estimated_frequency * (1 - a - c) / (n_user * (a - c)))
            frequency_variances[key] = estimated_frequency_variances + sample_variance
            
    return estimated_frequencies, frequency_variances


def group_perturb_phase2(data, key_space, epsilon):
    """对数据集中的每个<K,V>对进行扰动,然后计算每个组的均值和方差"""
    n_user = len(data)
    n_key = len(key_space)
    epsilon_1 = np.log((np.exp(epsilon) + 1) / 2)
    epsilon_2 = epsilon
    a = np.exp(epsilon_1) / (np.exp(epsilon_1) + n_key - 1)
    c = 1 / (np.exp(epsilon_1) + n_key - 1)
    p = np.exp(epsilon_2) / (np.exp(epsilon_2) + 1)

    # 对键值进行扰动
    perturbed_data = [pckv_grr(kv_pair, key_space, epsilon_1, p) for kv_pair in data]

    """服务器端"""
    # 聚合器端进行估计均值和方差
    key_counts = {key: 0 for key in key_space}
    positive_counts = {key: 0 for key in key_space}
    negative_counts = {key: 0 for key in key_space}

    for key, value in perturbed_data:
        key_counts[key] += 1
        if value == 1:
            positive_counts[key] += 1
        else:
            negative_counts[key] += 1

    # 计算估计频率，均值
    estimated_frequencies = {}
    estimated_means = {}

    for key in key_space:
        if key_counts[key] > 0:
            n1 = positive_counts[key]
            n2 = negative_counts[key]
            frequency = ((n1 + n2) / n_user - c) / (a - c)
            # 矫正频率估计，防止分母为零
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
            est = np.linalg.solve(A, b_n)  # 解方程 A * est = b_n

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
    group_counts = data[group_by_attributes].value_counts().to_dict()

    # 计算各组的采样概率： p_g = s_g / n_g
    prob_dict = {}
    for group, sample_size in sample_size_dict.items():
        n = group_counts[group]
        prob_dict[group] = (sample_size / n) if n > 0 else 0.0

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
        # 采样数据
        sampled = group_df.sample(n=count, random_state=random_state)
        sampled_list.append(sampled)
        # 剩余数据
        remaining = group_df.loc[~group_df.index.isin(sampled.index)]
        remaining_list.append(remaining)

    sampled_data = pd.concat(sampled_list, ignore_index=True)
    remaining_data = pd.concat(remaining_list, ignore_index=True)

    return sampled_data, remaining_data, group_sample_counts


def sample_allocation(estimated_frequency_variances, mb_2):
    """
    根据各组频率方差的平方根占比计算采样率和样本量
    """
    variance_sqrt = {key: np.sqrt(var) for key, var in estimated_frequency_variances.items()}
    sum_variance_sqrt = sum(variance_sqrt.values())
    
    # 计算各组的采样率
    sampling_rates = {key: sqrt_var/sum_variance_sqrt for key, sqrt_var in variance_sqrt.items()}
    
    sample_sizes = {key: int(rate * mb_2) for key, rate in sampling_rates.items()}
    
    total_allocated = sum(sample_sizes.values())
    remaining = mb_2 - total_allocated
    
    if remaining > 0:
        sorted_vars = sorted(estimated_frequency_variances.items(), key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            key = sorted_vars[i % len(sorted_vars)][0]
            sample_sizes[key] += 1
            
    return sample_sizes


if __name__ == '__main__':
    # 读取 CSV 文件
    file_path = "../../dataset/retirement/retirement_expanded.csv"
    table = pd.read_csv(file_path)

    # 定义分组查询所需参数
    group_by_attributes = ['Organization Group Code']  # 替换为实际的分组列名
    aggregation_attribute = 'Retirement'  # 替换为实际的聚合列名
    memory_budget = 63674  # 总样本容量= 1% * 用户总数
    alpha = 0.2  # phase1 和 phase2 分配的样本数比例

    # 计算phase1和phase2的样本数
    mb_1 = int(memory_budget * alpha)
    mb_2 = int(memory_budget * (1 - alpha))

    # 定义差分隐私参数
    epsilon = 2

    table_simple = table[[*group_by_attributes, aggregation_attribute]].copy()
    table_simple = table_simple.dropna()

    value_max = table[aggregation_attribute].max()
    value_min = table[aggregation_attribute].min()

    """在N1中均匀采样mb_1的数据量"""
    N1, N2, sample_size_1 = uniform_sampling(table_simple, group_by_attributes[0], mb_1)

    """N1数据预处理"""
    # 进行归一化
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    N1['value_normalized'] = scaler_1.fit_transform(N1[[aggregation_attribute]])

    """加噪后CV"""
    key_space = table_simple[group_by_attributes[0]].unique().tolist()
    n_key = len(key_space)

    simple_sizes_list = []

    data = list(N1[group_by_attributes + ['value_normalized']].itertuples(index=False, name=None))

    attempts = 0
    while len(simple_sizes_list) < 10 and attempts < 2000:
        try:
            # 扰动数据
            estimated_frequencies, estimated_frequency_variances = group_perturb_phase1(
                data, key_space, epsilon)

            # 检查方差估计中是否存在负值，由于浮点数精度问题，所以会存在负值方差
            if any(variance < 0 for variance in estimated_frequency_variances.values()):
                attempts += 1
                continue

            # 各组的采样量（不需要使用逆归一化均值和方差）
            simple_size = sample_allocation(estimated_frequency_variances, mb_2)

            if all(value > 0 for value in simple_size.values()):
                simple_sizes_list.append(simple_size)
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

    """阶段2"""
    """采样样本"""
    # 使用分层抽样
    sample_2 = stratified_sampling_probability(N2, average_simple_sizes, group_by_attributes[0])

    """对样本数据添加噪声"""
    # 进行归一化和离散化
    scaler_2 = MinMaxScaler(feature_range=(-1, 1))
    sample_2['value_normalized'] = scaler_2.fit_transform(sample_2[[aggregation_attribute]])
    sample_2['value_discretized'] = sample_2['value_normalized'].apply(discretize_value)

    sample_data = list(sample_2[group_by_attributes + ['value_discretized']].itertuples(index=False, name=None))

    real_result = table.groupby(group_by_attributes)[aggregation_attribute].agg('mean')
    print(f"各分组真实均值：{real_result}")

    relative_errors_list = []
    sample_means_list = []

    for i in range(10):
        try:
            sample_estimated_frequencies, sample_estimated_means = group_perturb_phase2(
                sample_data, key_space, epsilon)

            sample_restored_means = {key: scaler_2.inverse_transform([[m_k]]).item() for key, m_k
                                     in sample_estimated_means.items()}

            sample_means_list.append(sample_restored_means)

            sample_restored_means_series = pd.Series(sample_restored_means)
            relative_errors = (np.abs(sample_restored_means_series - real_result) / real_result).to_dict()
            relative_errors_list.append(relative_errors)

        except Exception as e:
            print(f"扰动样本数据失败: {e}")

    average_restored_means = pd.DataFrame(sample_means_list).mean().to_dict()
    print("样本估计均值：", average_restored_means)

    average_relative_errors = pd.DataFrame(relative_errors_list).mean().to_dict()
    print("相对误差：", average_relative_errors)

    average_all_relative_errors = np.mean(list(average_relative_errors.values()))
    print("所有组的平均相对误差:", average_all_relative_errors)

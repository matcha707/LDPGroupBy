import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def group_perturb(data, key_space, epsilon):
    """对数据集中的每个<K,V>对进行扰动,然后计算每个组的频率和方差"""
    n_user = len(data)
    n_key = len(key_space)
    epsilon_1 = np.log((np.exp(epsilon) + 1) / 2)
    epsilon_2 = epsilon
    a = np.exp(epsilon_1) / (np.exp(epsilon_1) + n_key - 1)
    b = 1 / (np.exp(epsilon_1) + n_key - 1)
    p = np.exp(epsilon_2) / (np.exp(epsilon_2) + 1)

    # 对键值进行扰动
    perturbed_data = [pckv_grr(kv_pair, key_space, epsilon_1, p) for kv_pair in data]

    """服务器端"""
    key_counts = {key: 0 for key in key_space}
    positive_counts = {key: 0 for key in key_space}
    negative_counts = {key: 0 for key in key_space}

    for key, value in perturbed_data:
        key_counts[key] += 1
        if value == 1:
            positive_counts[key] += 1
        else:
            negative_counts[key] += 1

    # 计算频率，均值
    estimated_frequencies = {}
    estimated_means = {}

    for key in key_space:
        if key_counts[key] > 0:
            n1 = positive_counts[key]
            n2 = negative_counts[key]

            frequency = ((n1 + n2) / n_user - b) / (a - b)
            # 矫正频率估计，防止分母为零
            estimated_frequency = min(max(frequency, 1 / n_user), 1)
            estimated_frequencies[key] = estimated_frequency

            # 计算各组的均值
            N = n_user * estimated_frequency
            # 生成矩阵A 和 向量 b_n
            A = np.array([
                [a * p - b/2, a * (1-p) - b/2],
                [a * (1-p) - b/2, a * p - b/2]
            ])
            b_n = np.array([n1, n2]) - n_user * b/2

            # 求解线性方程 A * est = b_n 得到 est (均值估计的向量)
            est = np.linalg.solve(A, b_n)  # 解方程 A * est = b_n

            # 限制 n1 和 n2 的估计值
            n1_est = min(max(float(est[0]), 1), N)
            n2_est = min(max(float(est[1]), 1), N)

            # 计算 m_k (均值)
            m_k = (n1_est - n2_est) / N

            estimated_means[key] = m_k

    return estimated_frequencies, estimated_means


# 对value_normalized进行离散化
def discretize_value(v):
    v = np.clip(v, -1, 1)
    prob = (1 + v) / 2
    return np.random.choice([1, -1], p=[prob, 1 - prob])


def stratified_sampling(data: pd.DataFrame, group_column: str, sample_size: int,
                        random_state: int = 42):
    """
    进行分层采样，根据每个组的频率分配样本量
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

    group_by_attributes = 'Organization Group Code'  # 替换为实际的分组列名
    aggregation_attribute = 'Retirement'  # 替换为实际的聚合列名

    # 定义隐私参数
    epsilon = 2

    """数据预处理"""
    memory_budget = 63674   # 用实际样本容量替换 1%
    table, group_sample_size = stratified_sampling(df, group_by_attributes, memory_budget)
    table = table[[group_by_attributes, aggregation_attribute]].copy()

    # 归一化
    scaler_1 = MinMaxScaler(feature_range=(-1, 1))
    table['value_normalized'] = scaler_1.fit_transform(table[[aggregation_attribute]])

    # 离散化
    table['value_discretized'] = table['value_normalized'].apply(discretize_value)

    key_space = df[group_by_attributes].unique().tolist()

    table_key_value = list(table[[group_by_attributes, 'value_discretized']].itertuples(index=False, name=None))

    n_iterations = 10
    all_means = {key: [] for key in key_space}
    all_relative_errors = {key: [] for key in key_space}
    completed_iterations = 0

    real_result = df.groupby(group_by_attributes)[aggregation_attribute].agg(['mean', 'var'])
    print("各组的真实均值和方差:", real_result)

    while completed_iterations < n_iterations:
        # 进行PCKV-GRR扰动
        estimated_frequencies, estimated_means = group_perturb(table_key_value, key_space, epsilon)

        # 使用逆归一化还原均值
        restored_means = {key: scaler_1.inverse_transform(np.array(m_k).reshape(-1, 1)).item() for key, m_k
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
    print("各组的估计均值:", average_means)

    average_relative_errors = {key: np.mean(errors) for key, errors in all_relative_errors.items()}
    print("相对误差:", average_relative_errors)

    average_all_relative_errors = np.mean(list(average_relative_errors.values()))
    print("所有组的平均相对误差:", average_all_relative_errors)


    

    



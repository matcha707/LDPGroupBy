import numpy as np
import pandas as pd
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


def duchi_mechanism(value, epsilon, domain=(-1, 1)):
    """对值应用Duchi机制"""
    def check_value(value, domain):
        if not domain[0] <= value <= domain[1]:
            raise ValueError("ERR: The input value={} is not in the input domain={}.".format(value, domain))
        return value

    if epsilon <= 0:
        raise ValueError("ERR: Epsilon must be positive.")

    p = np.exp(epsilon) / (np.exp(epsilon) + 1)

    value = check_value(value, domain)

    a, b = domain
    rnd_p = ((1 - 2 * p) * value + (a * p + b * p - a)) / (b - a)
    rnd = np.random.random()
    value = a if rnd <= rnd_p else b

    # 扰动值
    value = (value - (b + a) * (1 - p)) / (2 * p - 1)
    return value


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
        # 将剩余的样本分配给频率最高的组
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

    # 定义键空间
    key_space = df[group_by_attributes[0]].unique().tolist()

    """数据预处理"""
    # 随机抽取指定数量的数据
    memory_budget = 63674  # 用实际样本容量替换 1%
    # 均匀采样
    table, group_sample_size = stratified_sampling(df, group_by_attributes[0], memory_budget)
    print(f"各组均匀采样的样本量为：{group_sample_size}")

    # 定义隐私预算参数
    epsilon1 = 1
    epsilon2 = 1

    table = table[[group_by_attributes[0], aggregation_attribute]].copy()

    # 对value属性进行归一化处理到[-1, 1]范围
    scaler = MinMaxScaler(feature_range=(-1, 1))
    table['value_normalized'] = scaler.fit_transform(table[[aggregation_attribute]])

    # 应用GRR和Duchi扰动
    table['perturbed_group'] = table[group_by_attributes[0]].apply(lambda x: random_response(epsilon1, x, key_space))
    table['perturbed_value'] = table['value_normalized'].apply(lambda x: duchi_mechanism(x, epsilon2))

    # 恢复扰动后的聚合属性值到原始范围
    table['perturbed_value_original'] = scaler.inverse_transform(table[['perturbed_value']])

    # 根据扰动后的数据进行分组查询
    perturbed_grouped = table.groupby('perturbed_group')['perturbed_value_original'].mean()
    perturbed_grouped.name = 'perturbed_mean'

    print("扰动结果................")
    print(perturbed_grouped)

    # 直接对原始数据进行分组查询
    print("真实分组查询结果................")
    real_result = df.groupby(group_by_attributes[0])[aggregation_attribute].mean()
    real_result.name = 'true_mean'
    print(real_result)

    true_mean_dict = real_result.to_dict()
    perturbed_mean_dict = perturbed_grouped.to_dict()

    # 计算相对误差
    relative_errors = {}
    for key in true_mean_dict:
        if key in perturbed_mean_dict:
            true_mean = true_mean_dict[key]
            perturbed_mean = perturbed_mean_dict[key]
            relative_error = np.abs(true_mean - perturbed_mean) / np.abs(true_mean)
            relative_errors[key] = relative_error

    # 输出相对误差
    print("相对误差：")
    for key, error in relative_errors.items():
        print(f"{key}: {error}")

    # 计算平均相对误差
    average_relative_error = np.mean(list(relative_errors.values()))
    print(f"平均相对误差：{average_relative_error}")

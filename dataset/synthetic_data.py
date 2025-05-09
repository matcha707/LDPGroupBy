import numpy as np
from scipy.stats import beta, truncnorm
import os
import pandas as pd

def generate_power_law(alpha, size):
    """
    生成(0,1)区间内服从幂律分布的随机数
    """
    if alpha >= 1:
        raise ValueError("alpha必须小于1以确保分布在(0,1)区间内是有效的")
        
    u = np.random.uniform(0, 1, size=size)
    
    x = u**(1/(1-alpha))
    
    return x

def scale_to_range(values, min_val, max_val):
    """将[0,1]范围内的值缩放到[min_val, max_val]范围"""
    return min_val + values * (max_val - min_val)


if __name__ == "__main__":
    # group列表
    group_list = ['1', '2', '3', '4']

    result_dict = {}
    
    np.random.seed(42)
    
    # group 1: Beta
    result_dict['1'] = np.random.beta(5, 2, size=8000000)
    
    # group 2: Beta
    result_dict['2'] = np.random.beta(2, 8, size=5500000)

    # group 3: Gaussian
    mean, var = 0.5, 0.16
    std = np.sqrt(var)
    a, b = (0 - mean) / std, (1 - mean) / std
    trunc_gaussian = truncnorm(a, b, loc=mean, scale=std)
    gaussian_data = trunc_gaussian.rvs(size=2000000)
    result_dict['3'] = gaussian_data

    # group 4: Power-Law
    result_dict['4'] = generate_power_law(alpha=0.6, size=500000)

    data_list = []
    for group, values in result_dict.items():
        group_column = [group] * len(values)
        scaled_values = scale_to_range(values, 0, 1000)
        data_list.extend(zip(group_column, scaled_values))
    
    df = pd.DataFrame(data_list, columns=['group_num', 'agg_value_1'])
    
    # 添加第二个分组列 group_sex
    total_rows = len(df)
    male_mask = np.random.choice([True, False], size=total_rows, p=[0.7, 0.3])
    df['group_sex'] = np.where(male_mask, 'male', 'female')
    
    # 生成第二个聚合列 agg_value_2
    # 为male组生成Beta分布
    male_indices = df[df['group_sex'] == 'male'].index
    male_values = np.random.beta(6, 1, size=len(male_indices))
    male_values = scale_to_range(male_values, 0, 100)
    
    # 为female组生成Power-law分布
    female_indices = df[df['group_sex'] == 'female'].index
    female_values = generate_power_law(alpha=0.6, size=len(female_indices))
    female_values = scale_to_range(female_values, 0, 100)
    
    # 初始化agg_value_2列
    df['agg_value_2'] = 0.0
    df.loc[male_indices, 'agg_value_2'] = male_values
    df.loc[female_indices, 'agg_value_2'] = female_values
    
    save_dir = "../synthetic-data"
    os.makedirs(save_dir, exist_ok=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    save_path = os.path.join(save_dir, "synthetic_data.csv")
    df.to_csv(save_path, index=False)

    print(f"Data saved to: {save_path}")

    

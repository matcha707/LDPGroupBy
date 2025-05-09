import pandas as pd
import torch
import sdv
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import os

print("当前sdv版本:", sdv.__version__)
print("当前torch版本:", torch.__version__)
print("torch与cuda驱动是否兼容", torch.cuda.is_available())

print(torch.cuda.is_available())

# 读取原始数据
data = pd.read_csv('/data/dataset/retirement.csv')

# 创建并适配数据的元数据
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# 删除现有的元数据文件（如果存在）
if os.path.exists('metadata_retirement.json'):
    os.remove('metadata_retirement.json')

metadata.save_to_json('metadata_retirement.json')
print("元数据保存成功.........")

# 从json文件中加载元数据
loaded_metadata = SingleTableMetadata.load_from_json('metadata_retirement.json')

print("开始训练模型.........")
# 训练生成模型
synthesizer = CTGANSynthesizer(
    metadata=loaded_metadata,
    verbose=True,         # 打印训练过程的日志
    log_frequency=True    # 定期打印损失值等信息
)
synthesizer.fit(data)
print("模型训练成功.........")

# 生成新数据
synthetic_data = synthesizer.sample(4458840)  # 生成400万条数据
print("合成数据生成成功.........")

# 合并原始数据和生成的数据
expanded_data = pd.concat([data, synthetic_data], ignore_index=True)
print("数据合并成功.........")

# 保存扩充后的数据集
expanded_data.to_csv('/data/dataset/retirement_expended.csv', index=False)
print("数据保存成功.........")

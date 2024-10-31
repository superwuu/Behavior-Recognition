from tqdm import tqdm
import numpy as np 
import pickle
import torch
import pandas as pd
import os
from scipy import stats
# import random 
np.random.seed(42)

def npy2csv(input,output):
    res=np.load(input)

    max_indices = np.argmax(res, axis=1)

    # 创建DataFrame
    df = pd.DataFrame({
        'Index': range(len(max_indices)),
        'Value': max_indices
    })

    # 保存为CSV文件
    csv_file_path = output
    df.to_csv(csv_file_path, index=False)

# 定义目录路径 
directory_path = '/media/sdd/hsj/0competiton/Human_activity_recognition/1028res'

# 获取目录下的所有文件名
file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# 打印文件名
for file_name in file_names:
    out_name=file_name[:-3]+"csv"
    inpath=os.path.join(directory_path,file_name)
    outpath=os.path.join("/media/sdd/hsj/0competiton/Human_activity_recognition/1028csv",out_name)
    npy2csv(inpath,outpath)

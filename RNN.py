import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# 데이터 불러오기
df = pd.read_csv('202309.csv', sep=',')
# print(df.head())[:10]
#    1500000100  1500000200  1500000506  ...  1550383400  1550383500  1550383600
# 0        53.0        26.0        18.0  ...        21.0        18.0        26.0
# 1        31.0        28.0        19.0  ...        21.0        18.0        26.0
# 2        21.0        31.0        28.0  ...        21.0        18.0        26.0
# 3        24.0        38.0        21.0  ...        21.0        18.0        26.0
# 4        26.0        40.0        36.0  ...        21.0        18.0        26.0
timeseries = df.iloc[:, 0].values.astype('float32')[:500]
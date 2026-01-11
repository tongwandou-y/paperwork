# create_sliding_window_dataset完成了训练 (Train) 对齐
# ================== 用于从.mat文件加载数据 ==================

import dask.dataframe as dd # Dask库，类似于pandas，用于在大型数据集上进行并行计算。
import os                   # os模块用来与操作系统进行交互，如处理文件路径、环境变量访问等。
import numpy as np
import torch
from torch.utils.data import Dataset    # Dataset 是 PyTorch 中用来封装数据集的抽象基类，自定义的数据集都需要继承它。
import csv                  # Python 内置的处理 CSV 文件的模块，用于读写CSV文件
from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler用于对特征进行归一化处理，将数据缩放到给定的最小值与最大值之间（通常是0和1），计算公式为 (X - X_min) / (X_max - X_min)。

import scipy.io as spio

"""
create_sliding_window_dataset 函数：
    它接收失真信号 signal、理想信号 labels 和窗口大小 taps。
    它在 signal 上滑动一个长度为 taps 的窗口，这个窗口作为特征 (X)。
    它取这个窗口中心点对应的 labels 值作为标签 (Y)。
    它处理了信号和标签长度可能不完全匹配的边缘情况。
"""
def create_sliding_window_dataset(signal, labels, taps):
    """
    将一维时序信号转换为适用于DNN的滑动窗口数据集。
    输入:
        signal: 失真的信号序列 (numpy array)
        labels: 对应的理想信号序列 (numpy array)
        taps:   窗口大小 (均衡器抽头数)
    输出:
        X: 特征矩阵 (样本数, taps)
        Y: 标签向量 (样本数, 1)
    """
    X, Y = [], []
    # 延迟，使得窗口中心对应当前标签
    delay = taps // 2 # 整数除法
    
    # 确保信号长度足够
    if len(signal) < taps:
        print(f"警告：信号长度 ({len(signal)}) 小于抽头数 ({taps})，无法创建数据集。")
        return np.array(X), np.array(Y)
    
    # 遍历信号，创建窗口
    for i in range(len(signal) - taps + 1): # 这个循环完成了输入和标签的映射
        window = signal[i : i + taps]       # 输入
        X.append(window)
        # 标签对应窗口的中心点
        # 确保不会超出标签数组的边界
        if (i + delay) < len(labels):
            Y.append(labels[i + delay])     # 标签
        
    # 由于Y可能比X短，我们裁剪X以确保它们长度一致
    num_samples = len(Y)
    return np.array(X[:num_samples]), np.array(Y).reshape(-1, 1)    # reshape(-1, 1)确保标签是一个列向量


"""
代码段解释
for i in range(...): 会遍历一个数字序列。变量 i 会从 0 开始，依次递增。
    - len(signal): 获取 signal (失真信号) 数组的总长度。
    - len(signal) - taps: 这是滑动窗口可以开始的最后一个索引位置。例如，如果信号有100个点（len(signal)=100），窗口大小 taps 是7，那么最后一个窗口的起始位置 i 是 93（100 - 7），这个窗口会覆盖索引 93 到 99。
    - + 1: range(N) 函数生成的序列是从 0 到 N-1。为了让循环包含 len(signal) - taps 这个值（即最后一个合法的起始位置），我们需要加 1。
    - 总结: 这一行的意思是“遍历 signal 数组的每一个可能的窗口起始位置 i”。i 从 0 开始，一直到 len(signal) - taps 结束。
------------------------------------------
window = signal[i : i + taps]
    - signal[i : i + taps]: 数组切片。从signal数组中，提取一个从索引i开始，到索引i+taps-1结束的子数组（总共taps个元素）
    - window = ...: 将这个提取出来的子数组（即当前的滑动窗口）赋值给变量 window。
    - 总结: 这一行创建了当前迭代的特征窗口。
------------------------------------------
X.append(window)
    - X: 这是一个列表（List），用来存储所有的特征窗口。
    - .append(window): 将刚刚创建的 window 数组作为一个整体元素添加到列表 X 的末尾。
    - 总结: 这一行将当前的特征窗口保存到总的特征列表 X 中。
------------------------------------------
if (i + delay) < len(labels):
    - i + delay: 当前窗口 window（它从 i 开始）所对应的中心点在原始信号中的索引。
    - 检查我们要找的那个中心点标签是否存在于 labels 数组中。
"""
"""
关键的映射关系在 create_sliding_window_dataset 的循环中：
    输入 (X): window = signal[i : i + taps] (例如 signal[0:19])
    标签 (Y): Y.append(labels[i + delay]) (例如 labels[9])
这意味着，模型使用第0到18个失真符号（中心是第9个）来预测第9个理想符号。
这是一种标准的、完全正确的中心抽头对齐方式。
"""


class MatDataset(Dataset):
    """
    从 MATLAB 的 .mat 文件加载预处理好的训练/测试数据的数据集类。
    """
    def __init__(self, mat_file_path, seq_len, is_train=True):
        """
        构造函数。
        :param mat_file_path: .mat 文件的路径
        :param seq_len: 滑动窗口的半宽 (对应原始代码的 seq_len)
        :param is_train: 标志位，True表示加载训练数据，False表示加载测试数据
        """
        # --- 1. 参数定义 ---
        self.seq_len = seq_len
        self.is_train = is_train
        taps = 2 * self.seq_len + 1  # 滑动窗口的总长度 (taps)

        # --- 2. 加载数据 ---
        print(f"正在从 '{mat_file_path}' 加载数据...")
        mat_contents = spio.loadmat(mat_file_path)

        if self.is_train:
            # 加载训练数据和标签
            distorted_signal = mat_contents['dnn_train_input'].flatten()
            ideal_signal = mat_contents['dnn_train_label'].flatten()
            print("加载训练数据...")
        else:
            # 加载测试数据（标签是虚拟的，因为我们只做预测）
            distorted_signal = mat_contents['dnn_test_input'].flatten()
            ideal_signal = distorted_signal # 对于测试，标签可以暂时用自身替代，因为不会被用到
            print("加载测试数据...")

        # --- 3. 创建滑动窗口数据集 ---
        print(f"创建滑动窗口... (窗口总长度: {taps})")
        # 注意：这里的 'dnn_train_label' 已经是理想信号了
        # `create_sliding_window_dataset` 函数是我们新加的
        features, labels = create_sliding_window_dataset(distorted_signal, ideal_signal, taps)
        
        # --- 4. 转换为 PyTorch 张量 ---
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        
        print(f"数据准备完毕。特征尺寸: {self.features.shape}, 标签尺寸: {self.labels.shape}")

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.features)

    def __getitem__(self, index):
        # 根据索引获取一个样本
        # 在这个简单的DNN均衡器中，我们只需要输入特征和对应的标签
        # 我们返回三次是为了匹配 train.py 中 for 循环的格式 (a, c, after)
        # 对于这个DNN，'c'部分的数据没有用到，可以直接返回特征
        return self.features[index], self.features[index], self.labels[index]
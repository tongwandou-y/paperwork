# 此脚本用于测试输入到NN的文件Data_For_NN_PRBS23_train_-21.mat中样本和标签是否对齐了

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 1. 替换为你实际的 .mat 文件路径
mat_path = r'D:\paperwork\Experiment_Data\20Gsyms_20km\NN_Input_Data_mat\Data_For_NN_PRBS23_train_-21.mat'

print(f"正在加载数据: {mat_path} ...")
data = sio.loadmat(mat_path)

# 注意：请确保这里的键名与你实际保存的键名一致
rx_sig = data['dnn_train_input'].flatten()
tx_pam = data['dnn_train_label'].flatten()

# 取前 10000 个符号来计算偏移（避免计算量过大）
N_check = min(30000, len(rx_sig))
rx_sub = rx_sig[:N_check]
tx_sub = tx_pam[:N_check]

# 去均值
rx_sub = rx_sub - np.mean(rx_sub)
tx_sub = tx_sub - np.mean(tx_sub)

# 计算全局互相关
print("正在计算全局互相关...")
corr = np.correlate(rx_sub, tx_sub, mode='full')
lags = np.arange(-len(rx_sub) + 1, len(rx_sub))

# 找到相关性最大的偏移量
peak_idx = np.argmax(np.abs(corr))
best_lag = lags[peak_idx]
max_corr_val = corr[peak_idx] / (np.std(rx_sub) * np.std(tx_sub) * len(rx_sub))

print("\n" + "="*50)
print(f"【对齐诊断结果】")
print(f"最大皮尔逊相关系数: {max_corr_val:.4f}")
print(f"信号真实偏移量 (Lag): {best_lag} 个符号")
print("="*50 + "\n")

if best_lag == 0:
    print("结论：宏观上是对齐的。可能是 Python 数据集切片代码或归一化有 Bug。")
else:
    print(f"结论：MATLAB 数据严重错位！")
    if best_lag > 0:
        print(f"-> 接收信号 (Rx) 比 标签 (Tx) 滞后了 {best_lag} 个符号。")
        print(f"-> 修复方法：在 MATLAB 导出时，将 Rx 删掉前 {best_lag} 个点，Tx 删掉后 {best_lag} 个点。")
    else:
        print(f"-> 接收信号 (Rx) 比 标签 (Tx) 超前了 {abs(best_lag)} 个符号。")
        print(f"-> 修复方法：在 MATLAB 导出时，将 Tx 删掉前 {abs(best_lag)} 个点，Rx 删掉后 {abs(best_lag)} 个点。")

# 绘图展示
plt.plot(lags, np.abs(corr))
plt.title(f"Cross Correlation (Peak at Lag = {best_lag})")
plt.xlabel("Lag (symbols)")
plt.ylabel("Correlation Amplitude")
plt.grid(True)
plt.savefig("correlation_plot.png")
print("互相关图已保存为 correlation_plot.png")
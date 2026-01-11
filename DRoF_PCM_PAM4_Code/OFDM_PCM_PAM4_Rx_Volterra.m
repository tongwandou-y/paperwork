% =========================================================================
% DRoF - OFDM_PCM_PAM4_Rx_Volterra.m
% 目的：加载VPI波形，执行Volterra均衡，并计算最终BER
%
% 流程:
% 1. 基础设置 (加载测试集)
% 2. 加载测试参数 (PRBS23)
% 3. 加载测试VPI波形 (PRBS23)
% 4. 信号预处理 (同步, 定时恢复) -> 得到 test_input
% 5. 加载训练数据 (PRBS15) -> 得到 train_input, train_label
% 6. Volterra 均衡器 (训练 + 均衡)
% 7. 信号后处理 (PCM盲同步 + OFDM解调)
% 8. BER 计算
%==========================================================================
clc;clear;close all;

%% === 1. 基础设置 (Configuration) ===
% --- 仿真控制 ---
prbs_base_name = 'PRBS31';      % 目标评估序列: 'PRBS23' (用于测试集)
run_mode       = 'test';        % 运行模式: 'test' (通常接收端主要运行test模式)
model_type = 'Volterra';        % 模型类型 (用于文件名生成)
received_optical_power = -16;   % 设置接收光功率 (例如 -18)
suduandlength = '20Gsyms_30km_';% 速率和距离标识 (注意包含下划线)

vpi_data_path  = 'D:\paperwork\Experiment_Data\20Gsyms_30km\VPI_Output_Data_csv'; % VPI输出的.csv文件路径

% 参数文件所在的路径 (必须与发射端保存路径一致)
param_load_path = 'D:\paperwork\Experiment_Data\20Gsyms_30km\TX_Matlab_Param_mat';

% 图片保存文件夹
image_save_path = 'D:\paperwork\Experiment_Data\20Gsyms_30km\RX_Matlab_Result_Images_png\Volterra';
if ~exist(image_save_path, 'dir')
    mkdir(image_save_path);
end

% 结果报告保存路径
report_save_path = 'D:\paperwork\Experiment_Data\20Gsyms_30km\RX_Matlab_Result_Reports_txt\Volterra';
if ~exist(report_save_path, 'dir')
    mkdir(report_save_path);
end

%% === 2. 加载系统参数 (Load Parameters) ===
% 自动构建参数文件名(例如: DRoF_PCM_Parameters_PRBS23_test.mat)
param_filename = sprintf('DRoF_PCM_Parameters_%s_%s.mat', prbs_base_name, run_mode);

% 组合完整路径
param_full_path = fullfile(param_load_path, param_filename);

% 检查文件是否存在
if ~exist(param_full_path, 'file')
    error('错误: 未找到参数文件 \n%s \n请检查发射端脚本是否已运行并保存到正确位置。', param_full_path);
end

fprintf('正在加载系统参数: %s ...\n', param_full_path);
load(param_full_path);

% --- [2.1] OFDM 物理层参数 ---
M           = DRoF_PCM_Parameters.OFDM_M;           % QAM调制阶数
Rs          = DRoF_PCM_Parameters.OFDM_Rs;          % OFDM符号速率
fc          = DRoF_PCM_Parameters.OFDM_fc;          % 载波频率
Fs          = DRoF_PCM_Parameters.OFDM_Fs;          % 采样率
dt          = DRoF_PCM_Parameters.OFDM_dt;          % 采样时间间隔
Rate        = DRoF_PCM_Parameters.OFDM_Rate;        % 过采样率
N_Sc        = DRoF_PCM_Parameters.OFDM_N_Sc;        % 子载波数
SymPerCar   = DRoF_PCM_Parameters.OFDM_SymPerCar;   % 每个子载波符号数
CP          = DRoF_PCM_Parameters.OFDM_CP;          % 循环前缀比例
N_IFFT      = DRoF_PCM_Parameters.OFDM_N_IFFT;      % IFFT点数
Subcarrier  = DRoF_PCM_Parameters.OFDM_Subcarrier;  % 有效子载波索引
Rolloff     = DRoF_PCM_Parameters.OFDM_Rolloff;     % 滚降系数
Delay       = DRoF_PCM_Parameters.OFDM_Delay;       % 滤波器延迟
symLen      = DRoF_PCM_Parameters.OFDM_symLen;      % 符号长度序列
Lf          = DRoF_PCM_Parameters.OFDM_Lf;          % 带通下限
Uf          = DRoF_PCM_Parameters.OFDM_Uf;          % 带通上限

% --- [2.2] 理想发射数据 (用于BER/EVM计算) ---
Txdata      = DRoF_PCM_Parameters.OFDM_Txdata;      % 原始发送比特
base_QAM    = DRoF_PCM_Parameters.OFDM_base_QAM;    % 原始QAM符号 (用于EVM)
OFDM        = DRoF_PCM_Parameters.OFDM;             % 发送端理想OFDM模拟波形 (用于PCM参考)

% --- [2.3] PCM & PAM4 链路参数 ---
fb          = DRoF_PCM_Parameters.OFDM_fb;          % PCM采样率
R           = DRoF_PCM_Parameters.OFDM_R;           % 降采样倍数
quant       = DRoF_PCM_Parameters.PCM_quant;        % 量化位数
codebook    = DRoF_PCM_Parameters.PCM_codebook;     % 量化码本
MM          = DRoF_PCM_Parameters.PAM_MM;           % PAM阶数 (4)
Mm          = DRoF_PCM_Parameters.PAM_Mm;           % PAM比特数 (2)
PAM_code    = DRoF_PCM_Parameters.PAM_code;         % 发送端理想PAM符号标签 [0,1,2,3]

% --- [2.4] 硬件与DAC参数 ---
Fs_AWG      = DRoF_PCM_Parameters.Fs_AWG;           % AWG采样率 (64e9)
sps         = DRoF_PCM_Parameters.RRC_sps;          % 每个符号采样点数
R_AWG       = DRoF_PCM_Parameters.RRC_R_AWG;        % AWG上采样率
beta        = DRoF_PCM_Parameters.RRC_beta;         % RRC滚降系数
span        = DRoF_PCM_Parameters.RRC_N_span;       % RRC截断长度
FFE_Taps    = DRoF_PCM_Parameters.FFE_Taps;         % 发送端FFE抽头

disp('参数加载完成。');

%% === 3. 加载 VPI 接收波形 (Load VPI Waveform) ===
% 构造 VPI 输出的 CSV 文件名 (例如: Data_PRBS31_10Gsyms_10km_test_-21.csv)
vpi_csv_filename = sprintf('Data_%s_%s%s_%d.csv', prbs_base_name, suduandlength, run_mode, received_optical_power);
full_csv_path = fullfile(vpi_data_path, vpi_csv_filename);

if ~exist(full_csv_path, 'file')
    error('错误: 未找到波形文件 %s。\n请检查 VPI 导出路径。', full_csv_path);
end

fprintf('正在加载 VPI 波形: %s ...\n', full_csv_path);

% 读取 CSV (跳过前7行头信息，提取第2列数据)
header_lines = 7;
opts = detectImportOptions(full_csv_path, 'FileType', 'text', 'NumHeaderLines', header_lines);
vpi_data_matrix = readmatrix(full_csv_path, opts);

rx_raw = vpi_data_matrix(:, 2)'; % rx_raw 原始接收信号

% 基础预处理
rx_raw = [rx_raw 0];      % 补零 (防止后续索引越界)
rx_raw = rx_raw - mean(rx_raw); % 去直流
rx_raw = 0.5 * rx_raw / (sum(abs(rx_raw))/length(rx_raw)) + 1;
rx_raw = rx_raw / 2;

%% === 4. 信号预处理与同步 ===
fprintf('正在执行帧同步与定时恢复...\n');

% --- [4.1] 重建本地同步头 ---
% 必须与发射端逻辑完全一致，用于计算互相关
sync_head_filename = sprintf('%s_pam4.txt', prbs_base_name);
if ~exist(sync_head_filename, 'file')
    warning('警告: 同步头文件 %s 未找到，请确保该文件在当前目录下。', sync_filename);
end
synchead = load(sync_head_filename)';
synchead_seq = [synchead synchead]; % 重复一次 (Tx逻辑)

% 构建发射端完整前导: [00...00 (2000个), SyncHead, 0, 0]
preamble_tx = [zeros(1, 2000), synchead_seq, 0, 0];

% 模拟信道损伤前的波形 (FFE + RRC)
ffe_obj = serdes.FFE('Mode', 1, 'WaveType', 'Impulse', 'TapWeights', FFE_Taps);
preamble_ffe = ffe_obj(preamble_tx);
rrc_filter = rcosdesign(beta, span, sps);
preamble_upsampled = upsample(preamble_ffe, sps);
synch_waveform_ref = conv(preamble_upsampled, rrc_filter);

% --- [4.2] 帧同步 ---
[frame_idx, xcorr_val] = frameSyn(rx_raw, synch_waveform_ref);

% 绘制同步相关峰 (用于确认同步是否成功)
figure('Name', 'Frame Synchronization');
plot(xcorr_val); title(['Frame Sync Correlation (' run_mode ')']); grid on;

% --- [4.3] 截取有效数据 ---
% 计算有效数据起始点：帧头位置 + 前导长度
head_offset = (2000 + length(synchead_seq) + 2) * sps;
start_idx = frame_idx + head_offset;

% 计算理论需要的样本长度
total_pam_symbols = length(PAM_code);
total_samples_needed = total_pam_symbols * sps;
end_idx = start_idx + total_samples_needed - 1;

if end_idx > length(rx_raw)
    warning('接收信号长度不足，数据将被截断！(缺少 %d 点)', end_idx - length(rx_raw));
    rx_synced = rx_raw(1, start_idx : end); % rx_synced帧同步并截取后的信号
else
    rx_synced = rx_raw(1, start_idx : end_idx);
end
disp([run_mode, ' 数据流已成功同步并截取。']);


h_eye_raw = figure('Name', 'MATLAB - VPI Raw Signal Eye Diagram');
% 为了防止数据量过大导致卡顿，只取前 10000 个点画图
% sps*2 表示眼图显示的周期长度 (显示 2 个符号宽度)
eyediagram(rx_synced(1:min(10000, length(rx_synced))), sps*2);
title(['Rx Raw Eye Diagram (' run_mode ')']);

% ---------------- [新增] 自动保存原始眼图 ----------------
h_eye = gcf;
eye_filename = sprintf('Eye_%s_%s_%s_%ddBm.png', prbs_base_name, run_mode, model_type, received_optical_power);
full_eye_path = fullfile(image_save_path, eye_filename);
saveas(h_eye, full_eye_path);
fprintf('眼图已保存: %s\n', eye_filename);
% ----------------------------------------------------

disp('已生成原始眼图，请检查是否与 VPI 示波器一致。');




% --- [4.4] 低通滤波 (LPF) ---
fs = Fs_AWG;
BaudRate = fs / sps;
rx_lpf = LowPF(rx_synced, -BaudRate, BaudRate, Fs_AWG); % rx_lpf 低通滤波后的信号
rx_amp = abs(rx_lpf); % 取模

% --- [4.5] 定时恢复 ---
% 使用方差最大化准则寻找最佳采样相位
var_metrics = zeros(1, sps);
for i = 1:sps
    temp_samples = rx_amp(i : sps : end);
    if ~isempty(temp_samples)
        var_metrics(i) = var(temp_samples);
    end
end
[~, best_sample_phase] = max(var_metrics);
rx_down = rx_amp(best_sample_phase : sps : end); % rx_down 下采样后的信号

% --- [4.6] RMS归一化 ---
% 1. 去直流 (变为双极性信号)
rx_ac = rx_down - mean(rx_down);
% 2. RMS 功率归一化 (使信号功率为1，满足 Volterra 输入要求)
test_input = rx_ac / rms(rx_ac); % test_input 均衡器测试输入信号

% --- [4.7] 对齐理想标签 ---
% 确保 test_input 和 original_pam_test_data 长度严格一致
current_len = length(test_input);
if current_len < total_pam_symbols
    original_pam_test_data = PAM_code(1:current_len);
    test_input = test_input; % 保持原样
elseif current_len > total_pam_symbols
    test_input = test_input(1:total_pam_symbols);
    original_pam_test_data = PAM_code;
else
    original_pam_test_data = PAM_code;
end

fprintf('测试集预处理完成。有效样本数: %d\n', length(test_input));

%% === 5. 加载训练数据与构建训练集 ===
disp('----------------------------------------');
disp('正在处理训练波形 (Training Phase)...');

% --- [5.1] 加载训练集参数 (获取理想标签) ---
train_prbs_name = 'PRBS23';
train_mode_name = 'train';

% 构造文件名
train_param_filename = sprintf('DRoF_PCM_Parameters_%s_%s.mat', train_prbs_name, train_mode_name);

% 组合完整路径 (使用前面定义的 param_load_path)
train_param_full_path = fullfile(param_load_path, train_param_filename);

if ~exist(train_param_full_path, 'file')
    error('错误: 缺少训练参数文件 \n%s \n请先运行 Tx 脚本生成训练集 (run_mode="train")。', train_param_full_path);
end

% 加载参数并提取理想标签
train_params_struct = load(train_param_full_path);
train_PAM_code = train_params_struct.DRoF_PCM_Parameters.PAM_code; 
% 注意：train_PAM_code 是 [0, 1, 2, 3] 格式的整数序列

% --- [5.2] 加载训练集 VPI 波形 ---
train_csv_file = sprintf('Data_%s_%s%s_%d.csv', train_prbs_name, suduandlength, train_mode_name, received_optical_power);
train_full_path = fullfile(vpi_data_path, train_csv_file);

if ~exist(train_full_path, 'file')
    error('错误: 缺少训练波形文件 %s。', train_full_path);
end

fprintf('正在加载训练波形: %s ...\n', train_full_path);
opts_train = detectImportOptions(train_full_path, 'FileType', 'text', 'NumHeaderLines', 7);
vpi_train_matrix = readmatrix(train_full_path, opts_train);
rx_train_raw = vpi_train_matrix(:, 2)';

% 基础预处理 (补零 & 去直流)
rx_train_raw = [rx_train_raw 0];
rx_train_raw = rx_train_raw - mean(rx_train_raw);
rx_train_raw = 0.5 * rx_train_raw / (sum(abs(rx_train_raw))/length(rx_train_raw)) + 1;
rx_train_raw = rx_train_raw / 2;

% --- [5.3] 训练集同步 (Sync) ---
% 必须加载 PRBS15 专用的同步头
sync_head_train_file = sprintf('%s_pam4.txt', train_prbs_name);
if ~exist(sync_head_train_file, 'file')
    warning('缺少训练同步头文件: %s', sync_head_train_file);
end
synchead_train = load(sync_head_train_file)';
synchead_train = [synchead_train synchead_train]; % 重复一次

% 构建 PRBS15 参考前导 (FFE + RRC)
preamble_train_tx = [zeros(1,2000), synchead_train, 0, 0];
preamble_train_ffe = ffe_obj(preamble_train_tx); % 复用 Section 4 的 ffe_obj
synch_wave_train = conv(upsample(preamble_train_ffe, sps), rrc_filter); % 复用 rrc_filter

% 帧同步
[train_loc, ~] = frameSyn(rx_train_raw, synch_wave_train);

% 计算截取位置
train_head_offset = (2000 + length(synchead_train) + 2) * sps;
train_start = train_loc + train_head_offset;
train_len_needed = length(train_PAM_code) * sps;
train_end = train_start + train_len_needed - 1;

if train_end > length(rx_train_raw)
    rx_train_synced = rx_train_raw(train_start:end);
    warning('训练数据长度不足，将被截断。');
else
    rx_train_synced = rx_train_raw(train_start:train_end);
end

% --- [5.4] 滤波与下采样 (Filter & Downsample) ---
% 低通滤波
train_lpf = LowPF(rx_train_synced, -BaudRate, BaudRate, Fs_AWG);
train_amp = abs(train_lpf);

% 独立定时恢复 (必须针对训练集单独寻找最佳采样点)
metrics_train = zeros(1, sps);
for i = 1:sps
    temp_train = train_amp(i:sps:end);
 if ~isempty(temp_train)
  metrics_train(i) = var(temp_train); % 使用方差最大化准则
 end
end
[~, best_phase_train] = max(metrics_train);
% fprintf('训练集最佳采样相位: %d\n', best_phase_train);

% 下采样
train_down = train_amp(best_phase_train:sps:end);

% --- [5.5] 归一化与标签映射 ---
% 1. 输入归一化 (RMS = 1)
train_ac = train_down - mean(train_down);
raw_train_input = train_ac / rms(train_ac);

% 2. 标签归一化 (将 [0,1,2,3] 映射到 [-1, -0.33, 0.33, 1])
% 公式: (x - 1.5) / 1.5
% 这样做的目的是让 Label 的幅值范围和 Input 的幅值范围 (RMS=1) 大致匹配，利于 MSE 收敛
raw_train_label = (train_PAM_code - 1.5) / 1.5; 

% --- [5.6] 精细对齐 (Fine Alignment) ---
% 这一步非常重要！！！用于确保每一个训练样本都能与正确的标签进行对应
% 使用互相关解决微小的符号错位
disp('正在执行训练数据精细对齐...');
max_lag = 50;
min_len_train = min(length(raw_train_input), length(raw_train_label));

% 计算互相关
[acor, lag] = xcorr(raw_train_input(1:min_len_train), raw_train_label(1:min_len_train), max_lag);
[~, I] = max(abs(acor));
lag_diff = lag(I);

fprintf('>>> 检测到训练集时延偏差 (Lag): %d 符号\n', lag_diff);

% 根据 Lag 调整数据
if lag_diff > 0
    % Rx滞后: 截去Rx头部，截去Label尾部
    train_input = raw_train_input(1+lag_diff : end);
    train_label = raw_train_label(1 : end-lag_diff);
elseif lag_diff < 0
    % Rx超前: 截去Rx尾部，截去Label头部
    lag_diff_abs = abs(lag_diff);
    train_input = raw_train_input(1 : end-lag_diff_abs);
    train_label = raw_train_label(1+lag_diff_abs : end);
else
    % 完美对齐
    train_input = raw_train_input;
    train_label = raw_train_label;
end

% 最终强制长度对齐 (取交集)
final_len = min(length(train_input), length(train_label));
train_input = train_input(1:final_len);
train_label = train_label(1:final_len);

% 转为列向量 (方便矩阵乘法)
% train_input 和 train_label 就是喂给 Volterra 模型的最终数据
train_input = train_input(:);
train_label = train_label(:);

disp(['训练数据准备完毕。最终样本数: ', num2str(length(train_input))]);

%% === 6. Volterra 均衡器 (Volterra Equalizer) ===
disp('----------------------------------------');
disp('开始 Volterra 均衡器训练...');

% --- [6.1] 均衡器参数配置 ---
L_linear    = 41;   % 线性项记忆长度 (Linear Memory Length)
L_nonlinear = 11;   % 二阶非线性记忆长度 (2nd-order Memory Length)
lambda      = 1e-6; % 岭回归正则化系数，用于防止矩阵奇异或过拟合

fprintf('配置: 二阶 Volterra (Full Cross-terms), L1=%d, L2=%d, Lambda=%.1e\n', ...
        L_linear, L_nonlinear, lambda);

% --------------------------------------
% 【关键逻辑】中心对齐
%     - Volterra 均衡器由两部分组成：一个长窗口（线性）和一个短窗口（非线性）。
%     - 为了让这两部分共同估计同一个时刻的输出符号，它们的中心抽头必须对齐。
% --------------------------------------

% 计算中心抽头索引 (Center Tap Index)
% 用于对齐不同长度窗口的中心，防止时序错位
center_linear     = (L_linear + 1) / 2;	% 线性滤波器窗口的中心位置
center_non_linear = (L_nonlinear + 1) / 2; % 非线性滤波器窗口的中心位置

% 计算非线性窗口需要的偏移量 (Offset)
offset_idx = center_linear - center_non_linear;

% --- [6.2] 构建训练矩阵 H_train ---
% 计算有效训练样本数
N_train_samples = length(train_input) - (L_linear - 1);

% 截取对应的理想标签 (取线性窗口的中心位置作为参考点)
target_train = train_label(center_linear : center_linear + N_train_samples - 1);

% 1. 构建线性特征矩阵 (H1)
H1_train = zeros(N_train_samples, L_linear);
for i = 1:N_train_samples
    % 截取线性窗口: [x(n)...x(n+L1-1)]
    H1_train(i, :) = train_input(i : i + L_linear - 1).';
end

% 2. 构建二阶非线性特征矩阵 (H2)
num_2nd_terms = L_nonlinear * (L_nonlinear + 1) / 2;
H2_train = zeros(N_train_samples, num_2nd_terms);

for i = 1:N_train_samples
    % 截取非线性窗口 (加上 offset 以对齐中心)
    % 加上偏移量是为了非线性窗口能够取到信号的中心部分
    win_nl = train_input(i + offset_idx : i + offset_idx + L_nonlinear - 1);
    
    col_idx = 1;
    % 双重循环生成交叉项
    for j = 1:L_nonlinear
        for k = j:L_nonlinear % 从 j 开始，避免重复 (如 x1*x2 和 x2*x1 是一样的)
            H2_train(i, col_idx) = win_nl(j) * win_nl(k);
            col_idx = col_idx + 1;
        end
    end
end

% 3. 构建偏置项 (Bias)
% 二阶非线性项会产生直流分量。添加这一列是为了让均衡器能自动学习并抵消这个直流偏置
H_bias_train = ones(N_train_samples, 1);

% 组合总训练矩阵
H_train = [H1_train, H2_train, H_bias_train];

% --- [6.3] 求解系数 (Training) ---
R_mat = H_train' * H_train;
P_vec = H_train' * target_train; % target_train为理想标签
% 岭回归
Volterra_Coeffs = (R_mat + lambda * eye(size(R_mat))) \ P_vec; % Volterra_Coeffs 最优滤波器系数

% 验证
% 检查一下训练出来的均衡器在训练集上的表现，确保逻辑没崩
y_train_est = H_train * Volterra_Coeffs;
% 计算 MSE
mse_train = mean(abs(y_train_est - target_train).^2);
fprintf('训练完成。训练集 MSE: %.6f\n', mse_train);

% --------------------------------------
% 【关键逻辑】为什么测试集也要构造矩阵？
%     - 因为滤波器系数只认位置
%     - 把测试集构造成得到这些系数的矩阵的样子
%     - 才能正确的把这些系数应用的到测试集上，以进行均衡
% --------------------------------------

% --- [6.4] 应用于测试集 (Testing) ---
% 构建测试集的线性项
N_test_samples = length(test_input) - (L_linear - 1);

% 1. 测试集 H1
H1_test = zeros(N_test_samples, L_linear);
for i = 1:N_test_samples
    H1_test(i, :) = test_input(i : i + L_linear - 1).';
end

% 2. 测试集 H2
H2_test = zeros(N_test_samples, num_2nd_terms);
for i = 1:N_test_samples
    % 同样的 offset 对齐逻辑
    win_nl = test_input(i + offset_idx : i + offset_idx + L_nonlinear - 1);
    col_idx = 1;
    for j = 1:L_nonlinear
        for k = j:L_nonlinear
            H2_test(i, col_idx) = win_nl(j) * win_nl(k);
            col_idx = col_idx + 1;
        end
    end
end

% 3. 测试集 Bias
H_bias_test = ones(N_test_samples, 1);

% 组合测试矩阵
H_test = [H1_test, H2_test, H_bias_test];
% 应用训练好的系数    均衡输出！！！
received_eq = (H_test * Volterra_Coeffs).'; % 转置为行向量 
% received_eq就是最终均衡后的信号，消除了色散和非线性失真的PAM4符号序列。

disp('均衡完成。正在进行 PAM4 判决...');




%% === 【核心可视化】 均衡效果三联图 ===
% 这是一个非常有意义的诊断图，用于对比 Volterra 均衡前后的信号质量

h_perf = figure('Name', 'Volterra Equalizer Performance Check', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);

% 定义理想电平 (避免重复写)
ideal_levels = [-1, -0.333, 0.333, 1];

% --- 子图 1: 均衡前的散点图 (test_input) ---
subplot(1, 3, 1);
% 为了绘图清晰，只取前 3000 个点
limit_len = min(3000, length(test_input));
plot(test_input(1:limit_len), '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 4);
title('1. Input (Before Volterra)');
xlabel('Symbol Index'); ylabel('Amplitude');
grid on; axis tight; ylim([-2 2]);

% 【修改点 1】使用循环绘制参考线
for lvl = ideal_levels
    yline(lvl, 'r--', 'Alpha', 0.5);
end

% --- 子图 2: 均衡后的散点图 (received_eq) ---
subplot(1, 3, 2);
plot(received_eq(1:limit_len), 'b.', 'MarkerSize', 5);
title('2. Output (After Volterra)');
xlabel('Symbol Index'); ylabel('Amplitude');
grid on; axis tight; ylim([-2 2]);

% 【修改点 2】使用循环绘制参考线
for lvl = ideal_levels
    yline(lvl, 'r--', 'LineWidth', 1.5);
end
legend('Eq Output', 'Ideal Levels', 'Location', 'best');

% --- 子图 3: 均衡后的直方图 (Histogram) ---
% 直方图能更直观地看出“山峰”是否尖锐，山峰越尖锐，误码率越低
subplot(1, 3, 3);
histogram(received_eq, 100, 'FaceColor', 'b', 'EdgeColor', 'none');
title('3. Output Histogram');
xlabel('Amplitude'); ylabel('Count');
grid on; xlim([-1.5 1.5]);

% 【修改点 3】使用循环绘制垂直参考线 (xline)
for lvl = ideal_levels
    xline(lvl, 'r--', 'LineWidth', 1.5);
end

% ---------------- [新增] 自动保存Volterra性能检查图 ----------------
h_perf_fig = gcf;
perf_filename = sprintf('Volterra_Performance_%s_%s_%s_%ddBm.png', prbs_base_name, run_mode, model_type, received_optical_power);
full_perf_path = fullfile(image_save_path, perf_filename);
saveas(h_perf_fig, full_perf_path);
fprintf('Volterra性能检查图已保存: %s\n', perf_filename);
% -----------------------------------------------------------------

fprintf('>> 已生成均衡前后对比图。请观察图2中的蓝点是否紧密收敛在红线附近。\n');






%% === 7. 信号后处理与 PCM 恢复 ===
disp('----------------------------------------');
disp('正在进行 PAM4 判决与 PCM 盲同步...');

% --- [7.1] PAM4 判决 (Hard Decision) ---
% 我们的目标电平归一化到了 [-1, -0.333, 0.333, 1]
% 对应的判决符号: 0, 1, 2, 3
% 判决门限:
%   Th1 = (-1 + -0.333)/2 = -0.667 (-2/3)
%   Th2 = (-0.333 + 0.333)/2 = 0
%   Th3 = (0.333 + 1)/2 = 0.667 (2/3)

PAM_decisions = zeros(size(received_eq));

% 执行判决
PAM_decisions(received_eq < -2/3) = 0;
PAM_decisions(received_eq >= -2/3 & received_eq < 0) = 1;
PAM_decisions(received_eq >= 0 & received_eq < 2/3) = 2;
PAM_decisions(received_eq >= 2/3) = 3;

% --- 计算真实的 PAM4 SER (鲁棒对齐版) ---
max_lag_ser = 100; 
len_corr_ser = min(length(PAM_decisions), length(original_pam_test_data));

[acor_ser, lag_ser] = xcorr(PAM_decisions(1:len_corr_ser), original_pam_test_data(1:len_corr_ser), max_lag_ser);
[~, I_ser] = max(abs(acor_ser));
best_lag_ser = lag_ser(I_ser);

fprintf('>>> SER计算对齐: 检测到 Lag = %d\n', best_lag_ser);

if best_lag_ser > 0
    idx_pam_start = 1 + best_lag_ser;
    idx_ref_start = 1;
else
    idx_pam_start = 1;
    idx_ref_start = 1 + abs(best_lag_ser);
end

len_pam_avail = length(PAM_decisions) - idx_pam_start + 1;
len_ref_avail = length(original_pam_test_data) - idx_ref_start + 1;
len_common_ser = min(len_pam_avail, len_ref_avail);

pam_aligned = PAM_decisions(idx_pam_start : idx_pam_start + len_common_ser - 1);
ref_aligned = original_pam_test_data(idx_ref_start : idx_ref_start + len_common_ser - 1);

[~, ser_ratio] = symerr(ref_aligned, pam_aligned);


fprintf('物理层 SER (PAM4): %.4e\n', ser_ratio);

% --- [7.2] PCM 盲帧同步 ---
% 通过最小粗糙度法，寻找正确的切分位置，把PAM解调后得到的串形比特流按照量化比特位quant进行切分

% 原理: 错误的比特位移会导致重建的 PCM 信号出现剧烈跳变(粗糙度高)。
% 正确的位移恢复出来的信号是一个连续变化的模拟波形、波形很平滑
% 错误的位移恢复出来的像是杂乱的噪声、波形很粗糙
% 我们遍历所有可能的位移(0 ~ quant-1)，寻找"最平滑"的那个。

% 1. 符号转比特 (Gray Mapping: 0->00, 1->01, 2->11, 3->10)
% 注意: 这里必须与发射端 pammod 的映射方式一致
decoded_decimal = gray2bin(PAM_decisions, 'pam', MM); 
rx_bits_matrix = de2bi(decoded_decimal, Mm, 'left-msb')';
rx_bits_stream = rx_bits_matrix(:);

% 2. 盲搜索最佳偏移
best_offset = 0;
min_roughness = inf;
roughness_history = zeros(1, quant); % 记录以便观察

% 3. 暴力穷举
for shift = 0 : quant - 1
    % (a) 尝试截取
    temp_bits = rx_bits_stream(1 + shift : end);
    n_samples = floor(length(temp_bits) / quant);
    
    % 性能优化：仅使用前 1000 个点进行快速评估
    n_test = min(n_samples, 1000);
    if n_test < 100, continue; end
    
    temp_bits_test = temp_bits(1 : n_test * quant);
    
    % (b) 恢复 PCM 电压值
    temp_bin = reshape(temp_bits_test, quant, n_test)';
    temp_dec = bi2de(temp_bin, 'left-msb') + 1; % +1 适配 MATLAB 索引
    
    % 查表 (Codebook)
    temp_pcm_volts = codebook(temp_dec);
    
    % (c) 计算粗糙度 (Roughness Metric)
    % 差分绝对值的均值: mean(|x[n] - x[n-1]|)
    roughness = mean(abs(diff(temp_pcm_volts)));
    roughness_history(shift+1) = roughness;
    
    % (d) 锁定最佳位移：也就是记录粗糙度最小的那个shift，这个就是最佳比特偏移量
    if roughness < min_roughness
        min_roughness = roughness;
    best_offset = shift;
    end
end

fprintf('>>> 盲同步锁定: 最佳位移 = %d (最小粗糙度: %.4f)\n', best_offset, min_roughness);

% --- [7.3] 重构最终 PCM 信号 ---
% 使用找到的最佳位移进行全量恢复
final_bits = rx_bits_stream(1 + best_offset : end);
n_final_samples = floor(length(final_bits) / quant);
final_bits = final_bits(1 : n_final_samples * quant);

% 比特 -> 十进制 -> 电压
final_bin_mat = reshape(final_bits, quant, n_final_samples)';
final_dec_idx = bi2de(final_bin_mat, 'left-msb') + 1;
% 确保索引不越界
final_dec_idx(final_dec_idx > length(codebook)) = length(codebook);
final_dec_idx(final_dec_idx < 1) = 1;

pcm_recovered = codebook(final_dec_idx);
if isrow(pcm_recovered), pcm_recovered = pcm_recovered.'; end % 确保列向量

fprintf('PCM 信号已恢复，样本数: %d\n', length(pcm_recovered));

% --- [7.4] 计算 信号量化噪声比 SQNR ---
% SQNR用于衡量接收端恢复出来的 PCM 模拟波形(pcm_recovered)与发射端原始的理想波形(ref_pcm_tx)有多接近

% 需要将恢复的 PCM 信号与发射端的理想 OFDM 信号对齐

% 准备参考信号: OFDM (来自 Section 2, 需降采样到 PCM 速率)
ref_pcm_tx = OFDM(1 : R : end);
ref_pcm_tx = ref_pcm_tx(:);

% 计算互相关找对齐点
len_corr = min(length(pcm_recovered), length(ref_pcm_tx));
[xc, lags] = xcorr(pcm_recovered(1:len_corr), ref_pcm_tx(1:len_corr));
[~, max_idx] = max(abs(xc));
pcm_lag = lags(max_idx);

% --- SQNR 计算用的临时截取 (为了算分) ---
if pcm_lag > 0
    idx_rx_start = 1 + pcm_lag;
    idx_ref_start = 1;
elseif pcm_lag < 0
    idx_rx_start = 1;
    idx_ref_start = 1 + abs(pcm_lag);
else
    idx_rx_start = 1;
    idx_ref_start = 1;
end

len_rx_avail = length(pcm_recovered) - idx_rx_start + 1;
len_ref_avail = length(ref_pcm_tx) - idx_ref_start + 1;

if len_rx_avail <= 0 || len_ref_avail <= 0
    warning('SQNR 对齐失败，无法计算重叠区域。');
    sqnr_val = -Inf;
else
    len_common = min(len_rx_avail, len_ref_avail);
    sig_rec = pcm_recovered(idx_rx_start : idx_rx_start + len_common - 1);
    sig_ref = ref_pcm_tx(idx_ref_start : idx_ref_start + len_common - 1);
    
    noise_power = mean(abs(sig_ref - sig_rec).^2);
    signal_power = mean(abs(sig_rec).^2);
    sqnr_val = 10 * log10(signal_power / noise_power);
end

fprintf('----------------------------------------\n');
fprintf('Final PCM SQNR = %.2f dB\n', sqnr_val);
fprintf('----------------------------------------\n');


% --- [核心修改 3: 物理对齐修正] ---
% 必须修改 pcm_recovered 本身，以匹配 OFDM 帧头位置
if pcm_lag > 0
    % Rx 滞后: 切掉 Rx 前面的多余数据
    pcm_recovered = pcm_recovered(1+pcm_lag : end);
    fprintf('>>> 已修正 PCM 数据流偏移 (Lag: %d)，截断头部。\n', pcm_lag);
    
elseif pcm_lag < 0
    % Rx 超前 (Lag < 0): 说明 Rx 缺失了头部数据
    % 必须在 Rx 头部补零，将其"顶"回正确的时间位置
    padding_len = abs(pcm_lag);
    pcm_recovered = [zeros(padding_len, 1); pcm_recovered];
    fprintf('>>> 已修正 PCM 数据流偏移 (Lag: %d)，头部补零 %d 个点以恢复对齐。\n', pcm_lag, padding_len);
    
else
    fprintf('>>> PCM 数据流对齐完美 (Lag: 0)，无需调整。\n');
end


% --- [7.5] 构建 DA_code (用于 OFDM 解调) ---
% 上采样
% 通过插值，将低采样率的PCM恢复信号pcm_recovered恢复到高采样率Fs，以进行OFDM的恢复

% 注意：直接用 pcm_recovered (未对齐截断的)，让 OFDM 接收机自己去切 CP

DA_code = zeros(length(OFDM), 1); % 长度与发射端原始的高速信号 OFDM 一致

% 计算需要填入多少个PCM采样点 (模拟 DAC 的保持特性，实际上就是简单的上采样)
num_points_map = min(length(pcm_recovered), floor(length(DA_code)/R));

% --------------------------------------
% 输入：紧凑的低速数据 [V1, V2, V3...]
% 输出：稀疏的高速数据 [V1, 0, 0..., V2, 0, 0..., V3...]
% --------------------------------------

for k = 1:num_points_map
    % 简单的插值：每隔 R 个点填入一个值
    idx_start = R * (k - 1) + 1;
    DA_code(idx_start, 1) = pcm_recovered(k);
end

disp('DA_code 已构建，准备进入 OFDM 解调...');

%% === 8. OFDM 解调与性能评估 ===
disp('----------------------------------------');
disp('正在进行 OFDM 解调...');

% --- [8.1] 信号重建 ---
% 使用带通滤波器模拟 DAC 后重建的模拟信号
% 输入: DA_code (稀疏的脉冲序列)
% 输出: re_OFDM (连续模拟波形)
re_OFDM = BandPF(DA_code', Lf, Uf, Fs);

% --- [8.2] 帧结构对齐 ---
% 计算 OFDM 单个符号的物理长度 (采样点数)
% Length = (N_IFFT + CP_Len) * Upsample_Rate
one_symbol_len_samples = ((1 + CP) * N_IFFT) * Rate;

% 计算整帧的目标总长度
target_total_len = SymPerCar * one_symbol_len_samples;  % 理论上整帧数据的采样点数
current_len = length(re_OFDM);

% 强制长度匹配 (截断或补零)
if current_len < target_total_len
    padding = zeros(1, target_total_len - current_len);
    re_OFDM = [re_OFDM, padding];
elseif current_len > target_total_len
    re_OFDM = re_OFDM(1:target_total_len);
end

% --- [8.3] OFDM 接收机链路 ---

% 1. 串并转换 (S/P) -> [符号数, 符号长度]
% 矩阵的每一行代表一个OFDM符号的时域波形
Rx_matrix = zeros(SymPerCar, one_symbol_len_samples);
for k = 1:SymPerCar
    idx_start = (k-1) * one_symbol_len_samples + 1;
    idx_end   = k * one_symbol_len_samples;
    Rx_matrix(k, :) = re_OFDM(1, idx_start : idx_end);
end

% 2. 下变频
reCarrier_OFDM = zeros(SymPerCar, one_symbol_len_samples);
time_axis_sym = symLen * dt; % 预计算时间轴 (注意 symLen 是 params 加载进来的)
cos_carrier = cos(2 * pi * fc * time_axis_sym); % 载波

for k = 1:SymPerCar
    reCarrier_OFDM(k, :) = Rx_matrix(k, :) .* cos_carrier;
end

% 3. 匹配滤波与下采样
% 目标: 恢复出 FFT 前的时域样值 (长度 = N_IFFT * (1+CP))
target_fft_len = N_IFFT * (1 + CP);
match_OFDM = zeros(SymPerCar, target_fft_len);

for k = 1:SymPerCar
    % 调用 MatchFilter (内含卷积 + 下采样)
    % 注意：MatchFilter 输出长度通常会有拖尾
    temp_filtered = MatchFilter(reCarrier_OFDM(k, :), Fs, Rs, Rolloff, Delay);
    
    % 精确截取有效部分 (跳过滤波器群延迟 Delay)
    % 理论起始点: Delay + 1
    % 理论结束点: Delay + target_fft_len
    start_read = Delay + 1;
    end_read   = start_read + target_fft_len - 1;
    
    % 边界保护
    if end_read > length(temp_filtered)
        % 如果不够长，补零 (罕见情况)
        pad_len = end_read - length(temp_filtered);
        temp_filtered = [temp_filtered, zeros(1, pad_len)];
    end
    
    match_OFDM(k, :) = temp_filtered(start_read : end_read);
end

% 4. 去除循环前缀
% 假设 CP 在前
cp_len_samples = CP * N_IFFT;
payload_OFDM = match_OFDM(:, cp_len_samples+1 : end);

% 5. FFT 变换
FFT_out = fft(payload_OFDM, N_IFFT, 2);

% 6. 提取有效子载波 (Subcarrier Extraction)
% Subcarrier 是有效数据的索引
re_QAM_matrix = FFT_out(:, Subcarrier);

% 转为一维复数序列
rx_qam_raw = reshape(re_QAM_matrix.', 1, N_Sc * SymPerCar);

%% === 9. 星座图校正与最终评估 (Correction & Metrics) ===
disp('正在进行星座图校正 (LS Equalization)...');

% 准备参考信号 (Ideal Tx Symbols)
% 截取长度以防万一 (通常长度是相等的)
min_sym_len = min(length(base_QAM), length(rx_qam_raw));
rx_qam_final = rx_qam_raw(1:min_sym_len);
ref_qam_final = base_QAM(1:min_sym_len);

% --- [9.1] 复数域最小二乘校正 ---
% 计算最佳复数因子 H，使得 sum(|rx * H - tx|^2) 最小
% 这可以同时校正：1. 幅度衰减 (AGC)  2. 全局相位旋转 (Phase Rotation)
% 公式: H = (rx \cdot tx') / (rx \cdot rx')  (最小化误差)
% 或者简单的: H_est = mean(ref ./ rx) 在高信噪比下也行，但 LS 更准

% 使用除法求解 (斜杠算符在 MATLAB 中自动做最小二乘)
H_est = (rx_qam_final * rx_qam_final') \ (rx_qam_final * ref_qam_final');

% 应用校正
re_QAM_corrected = rx_qam_final * H_est;

% --- [9.2] 性能计算 ---

% 1. EVM 计算
evm_obj = comm.EVM;
rmsEVM = step(evm_obj, ref_qam_final.', re_QAM_corrected.');

% 2. BER 计算
% QAM 解调 (Hard Decision)
re_bits_mat = qamdemod(re_QAM_corrected, M, 'gray', 'OutputType', 'bit');
re_bits_stream = re_bits_mat(:);

% 理想比特流
tx_bits_mat = qamdemod(ref_qam_final, M, 'gray', 'OutputType', 'bit');
tx_bits_stream = tx_bits_mat(:);

[numErr, BER_val] = biterr(tx_bits_stream, re_bits_stream);

% --- [9.3] 绘图与报告 ---
h_const = figure('Name', 'OFDM Final Constellation');
plot(real(re_QAM_corrected), imag(re_QAM_corrected), 'b.', 'MarkerSize', 6); hold on;
plot(real(ref_qam_final), imag(ref_qam_final), 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);

title(sprintf('After Correction\nEVM: %.2f%%', rmsEVM));
legend('Rx Symbols', 'Ideal Tx');
grid on; axis equal; axis([-4 4 -4 4]);

% ---------------- 自动保存星座图 ----------------
h_const_fig = gcf;
const_filename = sprintf('Constellation_%s_%s_%s_%ddBm.png', prbs_base_name, run_mode, model_type, received_optical_power);
full_const_path = fullfile(image_save_path, const_filename);
saveas(h_const_fig, full_const_path);
fprintf('星座图已保存: %s\n', const_filename);
% -------------------------------------------------------


%% === FINAL REPORT 输出 & 自动保存 ===

% 1. 构造报告文件名
report_filename = sprintf('Report_%s_%s_%s_%ddBm.txt', prbs_base_name, run_mode, model_type, received_optical_power);
full_report_path = fullfile(report_save_path, report_filename);

% 2. 打开文件 (权限 'w' 表示覆盖写入)
fid = fopen(full_report_path, 'w');

% 3. 构造内容并同时输出到屏幕和文件
report_lines = {};
report_lines{end+1} = '========================================';
report_lines{end+1} = '        FINAL PERFORMANCE REPORT        ';
report_lines{end+1} = '========================================';
report_lines{end+1} = sprintf('Data Source   : %s (%s) %d dBm', prbs_base_name, run_mode, received_optical_power);
report_lines{end+1} = sprintf('Model Type    : %s', model_type); 
report_lines{end+1} = sprintf('SER (PAM4)    : %.4e', ser_ratio);
report_lines{end+1} = sprintf('Volterra L1/L2: %d / %d', L_linear, L_nonlinear);
report_lines{end+1} = sprintf('PCM SQNR      : %.4f dB', sqnr_val);
report_lines{end+1} = sprintf('rms EVM       : %.4f %%', rmsEVM);
report_lines{end+1} = sprintf('BER (16-QAM)  : %.4e', BER_val);
report_lines{end+1} = sprintf('Total Errors  : %d / %d bits', numErr, length(tx_bits_stream));
report_lines{end+1} = '========================================';
report_lines{end+1} = sprintf('Date          : %s', datestr(now)); 

% 4. 循环写入
for i = 1:length(report_lines)
    disp(report_lines{i});            % 打印到屏幕
    fprintf(fid, '%s\r\n', report_lines{i}); % 打印到txt文件
end

% 5. 关闭文件
fclose(fid);
fprintf('性能报告已自动保存到: %s\n', full_report_path);

disp('脚本运行结束。');




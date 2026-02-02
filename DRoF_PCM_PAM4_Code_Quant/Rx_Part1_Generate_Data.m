% =========================================================================
% DRoF - Rx_Part1_Generate_Data.m (Final: Symbol Recovery & Regression)
% 目的：处理 VPI 输出，进行同步、滤波和符号提取，为 DNN 准备训练/测试数据
%
% 核心策略：
% 1. 符号级恢复：按照物理符号速率 (20Gsym/s) 提取 PAM4 符号。
% 2. 最小二乘回归：自动消除接收信号的 DC 偏置和幅度缩放误差。
% 3. 为 Part 2 准备：输出纯净的符号流，分数倍 PCM 重采样将在 Part 2 进行。

% 流程:
% 1. 在 "基础设置" 中设置 prbs_base_name 和 run_mode ('train' 或 'test')
% 2. 自动加载对应的 .mat 参数 和 .csv 波形
% 3. 执行统一的同步、滤波、定时恢复
% 4. 根据 run_mode，自动保存 "训练" 或 "测试" 所需的数据
%==========================================================================
clc;clear;close all;

%% === 1. 基础设置 (TRAIN / TEST) ===
% ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
% 以后只需要修改本节中的变量即可
% ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
prbs_base_name = 'PRBS23'; % 例如: 'PRBS23' 或 'PRBS31'
run_mode = 'train';     % 设置为: 'train' 或 'test'
received_optical_power = -20;     % 设置接收光功率 (对应文件名中的数字，如 -10, -14, -18 等)
suduandlength = '20Gsyms_30km';  % 速率和距离标识 (注意包含下划线)

% 手动修改为量化比特数 (必须与发射端一致！)
quant = 4; 

% --- 路径设置 ---
data_dir = 'D:\paperwork\PRBS_Data';
% 根据 quant 自动定位实验数据文件夹
root_base_dir = sprintf('D:\\paperwork\\Experiment_Data\\Quant\\20Gsyms_30km_%dbit', quant);

% 定义子文件夹路径
vpi_data_path   = fullfile(root_base_dir, 'VPI_Output_Data_csv'); 
param_load_path = fullfile(root_base_dir, 'TX_Matlab_Param_mat');
mat_save_path   = fullfile(root_base_dir, 'NN_Input_Data_mat');

% 自动创建保存目录
if ~exist(mat_save_path, 'dir'), mkdir(mat_save_path); end

%% === 2. 加载参数 (根据 run_mode 自动处理) ===
if strcmpi(run_mode, 'train')
  disp(['模式: 训练 (TRAIN). 使用 ', prbs_base_name]);
elseif strcmpi(run_mode, 'test')
  disp(['模式: 测试 (TEST). 使用 ', prbs_base_name]);
else
  error("无效的 'run_mode'。请在脚本顶部设置为 'train' 或 'test'。");
end

% 构造参数文件名 (例如: DRoF_PCM_Parameters_PRBS23_6bit_train.mat)
param_filename = sprintf('DRoF_PCM_Parameters_%s_%dbit_%s.mat', prbs_base_name, quant, run_mode);

% 组合完整路径
param_full_path = fullfile(param_load_path, param_filename);
disp(['加载参数: ', param_full_path]);

% 检查文件是否存在
if ~exist(param_full_path, 'file')
  error('错误: 找不到参数文件 \n%s \n请检查发射端脚本是否已运行并保存到正确位置。', param_full_path);
end

load(param_full_path); % 加载完整路径

% 解包所有参数
M = DRoF_PCM_Parameters.OFDM_M;
Rs = DRoF_PCM_Parameters.OFDM_Rs;
fc = DRoF_PCM_Parameters.OFDM_fc;
Fs = DRoF_PCM_Parameters.OFDM_Fs;
dt = DRoF_PCM_Parameters.OFDM_dt;
Rate = DRoF_PCM_Parameters.OFDM_Rate;
N_Sc = DRoF_PCM_Parameters.OFDM_N_Sc;
SymPerCar = DRoF_PCM_Parameters.OFDM_SymPerCar;
CP = DRoF_PCM_Parameters.OFDM_CP;
N_IFFT = DRoF_PCM_Parameters.OFDM_N_IFFT;
Txdata = DRoF_PCM_Parameters.OFDM_Txdata;
base_QAM = DRoF_PCM_Parameters.OFDM_base_QAM;
Subcarrier = DRoF_PCM_Parameters.OFDM_Subcarrier;
Rolloff = DRoF_PCM_Parameters.OFDM_Rolloff;
Delay = DRoF_PCM_Parameters.OFDM_Delay;
symLen = DRoF_PCM_Parameters.OFDM_symLen;
Lf = DRoF_PCM_Parameters.OFDM_Lf;
Uf = DRoF_PCM_Parameters.OFDM_Uf;
OFDM = DRoF_PCM_Parameters.OFDM;
fb = DRoF_PCM_Parameters.OFDM_fb;
R = DRoF_PCM_Parameters.OFDM_R;
samplenumber = DRoF_PCM_Parameters.OFDM_samplenumber;
quant = DRoF_PCM_Parameters.PCM_quant;
codebook = DRoF_PCM_Parameters.PCM_codebook;
MM  = DRoF_PCM_Parameters.PAM_MM;
Mm  = DRoF_PCM_Parameters.PAM_Mm;
PAM_code  = DRoF_PCM_Parameters.PAM_code;
R_AWG  = DRoF_PCM_Parameters.RRC_R_AWG;
sps  = DRoF_PCM_Parameters.RRC_sps;
Fs_AWG = DRoF_PCM_Parameters.Fs_AWG;
beta = DRoF_PCM_Parameters.RRC_beta;
span = DRoF_PCM_Parameters.RRC_N_span;
FFE_Taps = DRoF_PCM_Parameters.FFE_Taps;

%% === 3. 加载VPI波形 (根据 run_mode 自动处理) ===
% 从VPI导出的文件中加载接收信号
% 构造VPI文件名 (例如: Data_PRBS23_20Gsyms_30km_6bit_train_-20.csv)
vpi_csv_filename = sprintf('Data_%s_%s_%dbit_%s_%d.csv', prbs_base_name, suduandlength, quant, run_mode, received_optical_power);
filename = fullfile(vpi_data_path, vpi_csv_filename);
disp(['加载VPI波形: ', filename]);

header_lines = 7; % 数据从第8行开始，所以跳过前7行 

% 使用 'HeaderLines' 选项来跳过文件头
opts = detectImportOptions(filename, 'FileType', 'text', 'NumHeaderLines', header_lines);
vpi_output = readmatrix(filename, opts);

received = vpi_output(:, 2)'; % VPI导出的第二列是信号幅度，' 是为了转置为行向量。
received = [received 0];

% 简单的去直流 (后续回归会进行精确校准)
received = received - mean(received); 

% 绘制眼图
eyediagram(received(1,30000:35000),sps);
title_str = sprintf('PAM4 Eye Diagram Before Equalization (%s %dbit)', upper(run_mode), quant);
title(title_str);

%% === 4. 信号处理与同步 ===
%% ========================= Resample  ==========================
fs = Fs_AWG;      % 接收端的采样率应与AWG的采样率一致

%% =================  De-synchronization code  ==================
% --- 步骤1: 定义与发射端匹配的最终信号参数 --
Baudrate_pam = fs / sps;  % PAM信号的波特率 (20 Gsym/s)

% --- 步骤2: 自动重新构建发射端的完整前导序列（包括zeros和FFE） ---
% 动态构建同步头文件名 (例如: 'PRBS23_pam4.txt')
sync_head_name = sprintf('%s_pam4.txt', prbs_base_name);
% 组合完整路径
sync_head_filename = fullfile(data_dir, sync_head_name);  
synchead = load(sync_head_filename)';       % 自动加载 PAM4 同步头，加载128个符号的PAM4同步序列
synchead = [synchead synchead];             % 组成256个符号的同步头，与发射端一致

preamble_symbols = [zeros(1,2000), synchead, 0, 0]; % 这是FFE的输入序列的一部分  发射端在同步头前加了2000个0

% --- 步骤3: 应用与发射端完全相同的FFE处理 ---
FFEMode = 1;
FFE = serdes.FFE('Mode', FFEMode, 'WaveType', 'Impulse', 'TapWeights', FFE_Taps); % 使用加载的 Taps
preamble_after_ffe = FFE(preamble_symbols); % 对包含前导0的同步序列应用FFE

% --- 步骤4: 对经过FFE的序列进行脉冲成形 ---
rrc_filter_rx = rcosdesign(beta, span, sps);% 创建与发射端完全相同的RRC滤波器
synch_waveform_upsampled = upsample(preamble_after_ffe, sps);   % 对FFE处理后的同步符号序列进行上采样和脉冲成形
synch_waveform = conv(synch_waveform_upsampled, rrc_filter_rx); % 使用'full'卷积

% --- 步骤5: 使用最终生成的参考波形进行帧同步 ---
[frameLoc, xcorrPlot] = frameSyn(abs(received), abs(synch_waveform)); % 使用 abs 增强鲁棒性
figure;plot(xcorrPlot);
title_str = sprintf('Frame Synchronization Correlation (PAM4) (%s)', upper(run_mode));
title(title_str);
xlabel('Sample Index');ylabel('Correlation Value');grid on;

% --- 步骤6: 精确计算并截取数据流 ---
head1 = frameLoc + (2000 + 2*128 + 2) * sps;  % 数据起始点 = 相关峰位置 + (2000个零点 + 256同步头 + 两个0符号)的长度对应的采样数
pam_data_symbols = length(PAM_code);	% 实际数据部分的PAM符号数
total_symbols_to_process = pam_data_symbols;
stream_length_samples = total_symbols_to_process * sps; % 计算需要截取的总样本点数
end_index = head1 + stream_length_samples - 1;  % end_index的值和VPI导出的.csv中的总数据量不一致是正常的！多出的部分直接舍弃，总样本点数总共就是stream_length_samples这么多个

if end_index > length(received)
  warning('计算的 end_index 超出接收信号总长度！可能由同步失败导致，将截取到末尾。');
  stream = received(1, head1 : end);
else
    stream = received(1, head1 : end_index);    % 截取包含训练序列和数据部分的信号流
end
disp([run_mode, ' 数据流已成功同步并截取。']);

%% ===================== PAM recovery (LowPF) ===========================
% 低通滤波以提取基带信号包络
streamFilter = LowPF(stream, -1*Baudrate_pam, Baudrate_pam, fs);
streamAmp = real(streamFilter); % 取实部

%% = 5. 回归归一化与符号提取 (Regression Scaling) =
% -------------------------------------------------------------------------
% 使用最小二乘回归
% 目的：
% 1. 自动寻找最佳的下采样相位 (Best Phase)。
% 2. 自动计算最佳增益 (alpha) 和偏置 (beta)，将接收信号完美映射到 [-1, 1] 区间。
% 3. 解决 3-bit/7-bit 可能出现的 DC 漂移和幅度比例失调问题。
% -------------------------------------------------------------------------

% 构造理想标签用于回归 (映射到 [-1, 1] 区间)
PAM_code_ideal_dnn = (PAM_code - 1.5) / 1.5;

fprintf('\n>>> 启动回归归一化扫描 (SPS=%d)...\n', sps);
best_mse = inf; 
best_phase = 1;
best_sig_final = [];

% 扫描 sps 个可能的采样相位，寻找均方误差 (MSE) 最小的那一个
for ph = 1:sps
  % 1. 下采样 (整数倍抽取)
  temp_down = streamAmp(1, ph:sps:end);
  
  % 2. 长度对齐 (用于计算相关和回归)
  L = min([50000, length(temp_down), length(PAM_code_ideal_dnn)]);
  rx_seg = temp_down(1:L);
  tx_seg = PAM_code_ideal_dnn(1:L);
  
  % 3. 寻找最佳微调延迟 (Lag)
  % 虽然前面做了帧同步，但下采样后可能还有 +/- 1 的符号误差
  [acor, lags] = xcorr(rx_seg - mean(rx_seg), tx_seg - mean(tx_seg));
  [~, I] = max(abs(acor)); 
  current_lag = lags(I);
  
  % 对齐信号
  if current_lag > 0
    rx_aligned = temp_down(1 + current_lag : end);
    tx_aligned = PAM_code_ideal_dnn;
  elseif current_lag < 0
    rx_aligned = temp_down;
    tx_aligned = PAM_code_ideal_dnn(1 + abs(current_lag) : end);
  else
    rx_aligned = temp_down;
    tx_aligned = PAM_code_ideal_dnn;
  end
  
  % 截断到相同长度以进行矩阵运算
  len_common = min(length(rx_aligned), length(tx_aligned));
  rx_aligned = rx_aligned(1:len_common);
  tx_aligned = tx_aligned(1:len_common);
  
  % 4. 最小二乘回归 (Least Squares)
  % 求解模型: y = alpha * x + beta
  % 其中 y 是理想标签，x 是接收信号
  X_ls = [rx_aligned', ones(len_common, 1)]; 
  Y_ls = tx_aligned';             
  
  theta = X_ls \ Y_ls; % 矩阵除法求解回归系数
  alpha = theta(1);    % 增益
  beta_bias = theta(2);% 偏置
  
  % 应用校准
  rx_calibrated = rx_aligned * alpha + beta_bias;
  
  % 计算 MSE
  mse = mean((rx_calibrated - tx_aligned).^2);
  
  % 更新最佳结果
  if mse < best_mse
    best_mse = mse;
    best_phase = ph;
    
    % 保存最终校准后的信号
    best_sig_final = rx_calibrated; 

    % 注意：这里我们找到了最佳的信号处理参数，实际上获得了纯净的符号流。
    % 此时的 best_sig_final 已经与 PAM_code_ideal_dnn 对齐。    
    % 保存对应的完整理想标签 (非常重要，Part 2 需要它来做 xcorr)
    % 这里我们保存原始的 PAM_code，而不是裁剪过的
    % 但是为了 DNN 训练，输入和标签长度必须一致
    % 所以我们先记录下校准后的 rx，最后再匹配长度
  end
end

fprintf('   最佳相位: %d, 最佳 MSE: %.6f\n', best_phase, best_mse);

% 最终数据准备
received_amp_final = best_sig_final'; % 确保是行向量

% 限幅 (防止离群点影响 DNN)
received_amp_final(received_amp_final > 1) = 1;
received_amp_final(received_amp_final < -1) = -1;

% 再次确保输入和标签长度严格一致 (以较短的为准)
len_final = min(length(received_amp_final), length(PAM_code_ideal_dnn));
received_amp_final = received_amp_final(1:len_final);
PAM_code_final_label = PAM_code_ideal_dnn(1:len_final);

% 对应的原始 PAM 符号 (用于测试集的 ser 计算)
original_pam_test_data = PAM_code(1:len_final);

fprintf('最终样本数: %d\n', len_final);

%% === 6. 保存数据给NN (根据 run_mode 自动切换) ===
disp_str = sprintf('%s %s 总符号数: %d', prbs_base_name, run_mode, len_final);
disp(disp_str);

% 构造输出文件名 (例如: Data_For_NN_PRBS31_8bit_test_-15.mat)
output_filename = sprintf('Data_For_NN_%s_%dbit_%s_%d.mat', prbs_base_name, quant, run_mode, received_optical_power);

% 2. 检查保存目录是否存在，不存在则创建
if ~exist(mat_save_path, 'dir')
    mkdir(mat_save_path);
    disp(['目录不存在，已自动创建: ', mat_save_path]);
end

% 3. 组合完整路径
full_output_filename = fullfile(mat_save_path, output_filename);

% --- 根据模式保存不同变量 ---
if strcmpi(run_mode, 'train')
  % --- 保存训练数据 ---
    dnn_train_input = received_amp_final; % 提取用于DNN训练的失真信号 (PRBS23) 
    dnn_train_label = PAM_code_final_label; % 提取训练标签 (PRBS23, 格式 [-1, ...])
  
  save(full_output_filename, 'dnn_train_input', 'dnn_train_label');
  disp(['训练数据已成功保存到: ', full_output_filename]);
  
elseif strcmpi(run_mode, 'test')
  % --- 保存测试数据 ---
    % 1. 提取用于DNN均衡的测试信号 (PRBS31)
    dnn_test_input = received_amp_final;
    
    % 2. 提取对应的完整理想标签
    % 保存原始的、未裁剪的 PAM_code (0,1,2,3 格式)
    % 这样 Rx_Part2 才能利用 xcorr 在全局范围内找到数据的真实位置，从而进行正确的分数倍重采样。
    original_pam_test_data = PAM_code; 

    % 使用 full_output_filename 进行保存
  save(full_output_filename, 'dnn_test_input', 'original_pam_test_data');
  disp(['测试数据已成功保存到: ', full_output_filename]);
end

disp('脚本运行结束。Rx_Part1 已提取 PAM4 符号流。');
disp('注意：PCM 帧对齐与分数倍重采样将在 Rx_Part2 中进行。');
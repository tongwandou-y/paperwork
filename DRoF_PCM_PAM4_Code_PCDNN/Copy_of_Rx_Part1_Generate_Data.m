% =========================================================================
% DRoF - Rx_Part1_Generate_Data.m
% 目的：处理 VPI 输出，为DNN准备训练或测试数据
%
% 流程:
% 1. 在 "基础设置" 中设置 prbs_base_name 和 run_mode ('train' 或 'test')
% 2. 自动加载对应的 .mat 参数 和 .csv 波形
% 3. 执行统一的同步、滤波、定时恢复
% 4. 根据 run_mode，自动保存 "训练" 或 "测试" 所需的数据
%==========================================================================
clc;clear;close all;

%% === 1. 基础设置 (TRAIN / TEST) ===
% ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
% 以后只需要修改本节中的两个变量即可
% ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
prbs_base_name = 'PRBS23'; % 例如: 'PRBS23' 或 'PRBS31'
run_mode = 'train';        % 设置为: 'train' 或 'test'
received_optical_power = -20;     % 设置接收光功率 (对应文件名中的数字，如 -10, -14, -18 等)
suduandlength = '20Gsyms_20km';  % 速率和距离标识 (注意包含下划线)
quant = 8; % 手动修改为量化比特数 (必须与发射端一致！)

% --- 存放 PRBS txt 文件的目录 --- (必须与发射端一致！)
data_dir = 'D:\paperwork\PRBS_Data';

root_base_dir = 'D:\paperwork\Experiment_Data\20Gsyms_20km';

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

% 构造参数文件名 (例如: DRoF_PCM_Parameters_PRBS23_train.mat)
param_filename = sprintf('DRoF_PCM_Parameters_%s_%s.mat', prbs_base_name, run_mode);

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
bp_OFDM = DRoF_PCM_Parameters.PCM_bp_OFDM; % 量化前PCM带通采样点(用于PCM回归标签)
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
% 构造VPI文件名
% 格式: Data_PRBS31_20Gsyms_20km_test_-15.csv
vpi_csv_filename = sprintf('Data_%s_%s_%s_%d.csv', prbs_base_name, suduandlength, run_mode, received_optical_power);

filename = fullfile(vpi_data_path, vpi_csv_filename);
disp(['加载VPI波形: ', filename]);

header_lines = 7; % 数据从第8行开始，所以跳过前7行 

% 使用 'HeaderLines' 选项来跳过文件头
opts = detectImportOptions(filename, 'FileType', 'text', 'NumHeaderLines', header_lines);
vpi_output = readmatrix(filename, opts);

received = vpi_output(:, 2)'; % VPI导出的第二列是信号幅度，' 是为了转置为行向量。
received = [received 0];

% Normailize
% -----------------------------------------------------------------
received = received-mean(received); % 处理后波形在纵轴以0为中心对称
received = 0.5*received/(sum(abs(received))/length(received))+1;    % 处理后波形在纵轴以1为中心对称
received = received/2;  % 处理后波形在纵轴以0.5为中心对称
% -----------------------------------------------------------------
eyediagram(received(1,30000:35000),sps);
title_str = sprintf('PAM4 Eye Diagram Before Equalization (%s Data)', upper(run_mode));
title(title_str);


%% === 4. 信号处理与同步 ===
%% ========================= Resample  ==========================
fs = Fs_AWG;      % 接收端的采样率应与AWG的采样率一致

%% =================  Signal Extraction (No Sync)  ==================
% --- 步骤1: 定义信号参数 ---
Baudrate_pam = fs / sps;  % PAM信号的波特率

% --- 步骤2: 计算起始点 (重点) ---
% 发射端使用 conv(up_PAM, RRCtx) 会引入固定的滤波器群延迟。
% RRC滤波器的延迟采样点数为 (span * sps) / 2。
% 索引从1开始，因此理论上的数据起始点如下：
tx_filter_delay = (span * sps) / 2; 
head1 = tx_filter_delay + 1;

% 注意：如果你的VPI仿真链路（如光纤、器件）还引入了额外的物理延迟，
% 且VPI没有自动对齐，你需要在这里手动加上 VPI_Delay_Samples：
% head1 = tx_filter_delay + 1 + VPI_Delay_Samples;

% --- 步骤3: 计算并截取数据流 ---
pam_data_symbols = length(PAM_code);       % 实际数据部分的PAM符号数
stream_length_samples = pam_data_symbols * sps; % 计算需要截取的总样本点数
end_index = head1 + stream_length_samples - 1;  

% 检查截取是否越界
if end_index > length(received)
    warning('计算的 end_index 超出接收信号总长度！可能存在未计算的延迟或截断。');
    stream = received(1, head1 : end);
else
    stream = received(1, head1 : end_index);    % 截取纯数据部分的信号流
end
disp([run_mode, ' 数据流已成功截取（无同步头直接对齐）。']);

%% ===================== PAM recovery ===========================
% 低通滤波以提取基带信号包络
streamFilter = LowPF(stream, -1*Baudrate_pam, Baudrate_pam, fs);
streamAmp = abs(streamFilter);

% --- 定时恢复 (最大方差法) ---
metrics = zeros(1, sps); % 用于存储每个相位的评估指标（方差）
for i = 1:sps
    receivedAmp = streamAmp(1, i:sps:end);  % 下采样
    if ~isempty(receivedAmp)
        metrics(i) = var(receivedAmp); % 计算方差作为评估指标
    else
        metrics(i) = 0; % 如果同步失败，receivedAmp可能为空
    end
end
[~, SamIndex] = max(metrics); % 找到方差最大的采样相位

receivedAmp1 = streamAmp(1, SamIndex:sps:end); % 在最佳相位 SamIndex 下采样

%% ===================== 准备DNN的输入和标签 ===========================
% 1. 归一化失真信号 (输入)
% 这个归一化假设信号电平大致在[0, 1]范围, 映射到[-1, 1]
received_amp_norm = receivedAmp1*2 - 1;

% 2. 归一化理想标签 (根据模式不同)
if strcmpi(run_mode, 'train')
    % 训练模式: 
    % 	发射端 PAM_code 是 [0, 1, 3, 2] (格雷码序列)
    % 	DNN 输出层是 tanh, 范围 [-1, 1]
    % 	我们需要将标签映射到 [-1, -1/3, 1/3, 1]
    PAM_code_ideal_dnn = (PAM_code - 1.5) / 1.5;
else
    % 测试模式: 不需要生成此标签
    % PAM_code_ideal_dnn = (PAM_code - 1.5) / 1.5; % 测试不需要这个
end

% --------------------------------------------------------
% 标签映射关系 (格雷码):
% 当 PAM_code 里的符号是 0：label = (0 - 1.5) / 1.5 = -1
% 当 PAM_code 里的符号是 1：label = (1 - 1.5) / 1.5 = -1/3
% 当 PAM_code 里的符号是 2：label = (2 - 1.5) / 1.5 = +1/3
% 当 PAM_code 里的符号是 3：label = (3 - 1.5) / 1.5 = +1
% 所以，DNN被正确地训练了（它遵循了格雷码的电平顺序）：
%     符号 0 -> 输出 -1
%     符号 1 -> 输出 -1/3
%     符号 2 -> 输出 +1/3
%     符号 3 -> 输出 +1
% --------------------------------------------------------

%% === 5. 保存数据给NN (根据 run_mode 自动切换) ===

% 确保接收到的符号数和理想符号数一致
total_symbols = length(PAM_code); %
if length(received_amp_norm) > total_symbols
    received_amp_norm = received_amp_norm(1:total_symbols);
elseif length(received_amp_norm) < total_symbols
    warning('接收到的符号数少于预期，将裁剪理想标签以匹配。');
    total_symbols = length(received_amp_norm); % 以接收到的为准
    PAM_code = PAM_code(1:total_symbols); % 裁剪 [0,1,2,3] 格式的
    if strcmpi(run_mode, 'train')
        PAM_code_ideal_dnn = PAM_code_ideal_dnn(1:total_symbols); % 裁剪 [-1,...] 格式的
    end
end

disp_str = sprintf('%s %s 总符号数: %d', prbs_base_name, run_mode, total_symbols);
disp(disp_str);

% === [新增] 块级对齐：每个PCM采样对应 G 个PAM符号 ===
G = quant / Mm;
if mod(quant, Mm) ~= 0
    error('quant (%d) 不能被 Mm (%d) 整除，无法构造物理块对齐。', quant, Mm);
end

% 以PCM采样长度为上限，确保完整物理块对齐
num_blocks = floor(total_symbols / G);
num_blocks = min(num_blocks, length(bp_OFDM));
aligned_total_symbols = num_blocks * G;

if aligned_total_symbols < total_symbols
    received_amp_norm = received_amp_norm(1:aligned_total_symbols);
    PAM_code = PAM_code(1:aligned_total_symbols);
    if strcmpi(run_mode, 'train')
        PAM_code_ideal_dnn = PAM_code_ideal_dnn(1:aligned_total_symbols);
    end
end

% PCM回归标签（量化前电压），按物理块对齐
pcm_ref = bp_OFDM(1:num_blocks);
pcm_ref_max = max(abs(pcm_ref));
if pcm_ref_max > 0
    pcm_ref_norm = pcm_ref / pcm_ref_max; % 归一化到[-1, 1]
else
    pcm_ref_norm = zeros(size(pcm_ref));
end

% 构造输出文件名 (例如: Data_For_NN_PRBS31_test_-15.mat)
output_filename = sprintf('Data_For_NN_%s_%s_%d.mat', prbs_base_name, run_mode, received_optical_power);

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
    dnn_train_input = received_amp_norm; % 提取用于DNN训练的失真信号 (PRBS23 - 100%)
    dnn_train_label = PAM_code_ideal_dnn; % 提取训练标签 (PRBS23 - 100%, 格式 [-1, -1/3, 1/3, 1])
    dnn_train_pcm_ref = pcm_ref_norm; % 量化前PCM电压标签(块级对齐)
    
    save(full_output_filename, 'dnn_train_input', 'dnn_train_label', 'dnn_train_pcm_ref');
    disp(['训练数据已成功保存到: ', full_output_filename]);

elseif strcmpi(run_mode, 'test')
    % --- 保存测试数据 ---
    % 1. 提取用于DNN均衡的测试信号 (PRBS31 - 100%)
    dnn_test_input = received_amp_norm;
    % 2. 提取对应的完整理想标签 (PRBS31 - 100%, 格式 [0, 1, 2, 3])
    %    这用于 Rx_Part2_PostProcessing_test.m 中计算 PAM SER 和进行最终 16-QAM BER 对比
    % [0, 1, 2, 3]是刻意为之！！！后面如果不对再改
    original_pam_test_data = PAM_code; 
    dnn_test_pcm_ref = pcm_ref_norm; % 量化前PCM电压标签(块级对齐)

    % 修改处：使用 full_output_filename 进行保存
    save(full_output_filename, 'dnn_test_input', 'original_pam_test_data', 'dnn_test_pcm_ref');
    disp(['测试数据已成功保存到: ', full_output_filename]);
end

disp('现在可以去Pytorch中进行训练和均衡了。');
% =========================================================================
% 用于处理不同量化比特的实验，但目前脚本有bug，仍未解决奇数比特比偶数比特差的问题
% 主要做的修改有
% 1. 废弃固定的 R=4 抽取逻辑。
% 2. 引入 resample 函数，根据 quant 动态调整 PCM 采样率到 Fs 的转换比率。
% 3. 解决奇数比特 (如 7-bit) 因 3.5 符号周期导致的时序滑移问题。
% 加上模拟FEC后有效果，但是是否能加？是否能加在接受端仍然存疑，所以暂时先注释掉
%==========================================================================

% =========================================================================
% DRoF - Rx_Part2_PostProcessing_test.m
% 目的：加载DNN均衡后的信号，通过【分数倍重采样】恢复OFDM波形，并评估性能
%
% 核心修复：
% 1. 废弃固定的 R=4 抽取逻辑。
% 2. 引入 resample 函数，根据 quant 动态调整 PCM 采样率到 Fs 的转换比率。
% 3. 解决奇数比特 (如 7-bit) 因 3.5 符号周期导致的时序滑移问题。
%
% 流程:
% 1. 加载 PRBS31_test.mat (参数)
% 2. 加载 Data_For_NN_PRBS31_test.mat (理想标签)
% 3. 加载 NN_Output_test.mat (均衡结果)
% 4. 执行 PAM4 判决, SER, PCM 解码
% 5. [关键] 执行分数倍重采样 (Resample) 重建 OFDM 波形
% 6. OFDM 解调, 最终 BER 计算
%==========================================================================
clc;clear;close all;

%% === 1. 核心设置 (TEST) ===
% ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
% 以后只需要修改本节中的四个变量即可
% ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
% 确保这里的设置与你运行 OFDM_PCM_OOKtx_auto.m (test模式) 时一致
prbs_base_name = 'PRBS31';
run_mode = 'test';
received_optical_power = -20;  % 设置要分析的接收光功率
quant = 12; % 手动修改为量化比特数 (必须与发射端一致！)

% 模型类型选择: 'DNN' 或 'CNN'
% 必须与 Python configs.py 中的 config.model_type 保持一致！
model_type = 'DNN';

% --- 路径设置 (自动适配 quant) ---
root_base_dir = sprintf('D:\\paperwork\\Experiment_Data\\Quant\\20Gsyms_30km_%dbit', quant);

% [路径1] 理想标签所在的文件夹 (Data_For_NN_...mat)
mat_input_path = fullfile(root_base_dir, 'NN_Input_Data_mat');
% [路径2] Python输出结果所在的文件夹 (NN_Output_...mat)
mat_output_path = fullfile(root_base_dir, 'NN_Output_Data_mat');
% [路径3] 参数文件所在的文件夹 (必须与发射端保存路径一致)
param_load_path = fullfile(root_base_dir, 'TX_Matlab_Param_mat');
% [路径4] 图片保存文件夹
image_save_path = fullfile(root_base_dir, 'RX_Matlab_Result_Images_png', 'NN');
if ~exist(image_save_path, 'dir'), mkdir(image_save_path); end
% [路径5] 结果报告保存路径
report_save_path = fullfile(root_base_dir, 'RX_Matlab_Result_Reports_txt', 'NN');
if ~exist(report_save_path, 'dir'), mkdir(report_save_path); end

disp(['模式: 评估 (TEST). 使用 ', prbs_base_name]);

%% === 2. 加载参数 (TEST) ===
% 构造参数文件名 (例如: DRoF_PCM_Parameters_PRBS31_8bit_test.mat)
param_filename = sprintf('DRoF_PCM_Parameters_%s_%dbit_%s.mat', prbs_base_name, quant, run_mode);
% 组合完整路径
param_full_path = fullfile(param_load_path, param_filename);
disp(['加载参数: ', param_full_path]);

% 检查文件是否存在
if ~exist(param_full_path, 'file')
  error('错误: 找不到参数文件 \n%s \n请检查发射端脚本是否已运行。', param_full_path);
end
load(param_full_path);

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
codebook = DRoF_PCM_Parameters.PCM_codebook;
MM  = DRoF_PCM_Parameters.PAM_MM;
Mm  = DRoF_PCM_Parameters.PAM_Mm;
R_AWG  = DRoF_PCM_Parameters.RRC_R_AWG;
sps  = DRoF_PCM_Parameters.RRC_sps;
Fs_AWG = DRoF_PCM_Parameters.Fs_AWG;
beta = DRoF_PCM_Parameters.RRC_beta;
span = DRoF_PCM_Parameters.RRC_N_span;

% [补充计算] 物理层符号速率 (Gsym/s)
% 这是计算重采样率的关键
Rs_PAM_final = Fs_AWG / sps; 

%% === 3. 加载理想标签和DNN均衡结果 (TEST) ===
% -------------------------------------------------------------------------
% 1. 加载理想测试标签 (来自 MATLAB Part 1)
% -------------------------------------------------------------------------
% 格式例如: Data_For_NN_PRBS23_8bit_test_-18.mat
label_filename = sprintf('Data_For_NN_%s_%dbit_%s_%d.mat', prbs_base_name, quant, run_mode, received_optical_power);
full_label_path = fullfile(mat_input_path, label_filename);

disp(['加载理想标签: ', full_label_path]);
if ~exist(full_label_path, 'file')
  error(['找不到文件: ', full_label_path, ' 请检查是否已运行 Part 1。']);
end
% 仅加载 'original_pam_test_data' (格式 [0,1,2,3])
load(full_label_path, 'original_pam_test_data');
% 强制转为行向量
original_pam_test_data = original_pam_test_data(:).';

% -------------------------------------------------------------------------
% 2. 加载DNN均衡结果 (来自 Python)
% -------------------------------------------------------------------------
% 按照您指定的格式构造文件名: NN_Output_test_DNN_8bit_-27.mat
% 对应格式: NN_Output_{run_mode}_{model_type}_{quant}_{power}.mat
dnn_output_filename = sprintf('NN_Output_%s_%s_%dbit_%d.mat', run_mode, model_type, quant, received_optical_power);
dnn_output_full_path = fullfile(mat_output_path, dnn_output_filename);

disp(['加载DNN均衡结果: ', dnn_output_full_path]);
if ~exist(dnn_output_full_path, 'file')
  error(['找不到文件: ', dnn_output_full_path, ' 请检查Python脚本是否已运行。']);
end
disp(['加载DNN均衡结果: ', dnn_output_filename]);
load(dnn_output_full_path); % 加载一个名为 received_eq 的变量

%% ==================== 信号格式化及参数自动推断 ======================
% 确保数据是行向量
if size(received_eq, 1) > 1
  received_eq = received_eq.';
end

% 显示均衡后的眼图
eyediagram(received_eq(1, 1:min(2000, length(received_eq))), 2*sps/Mm); % PAM4眼图符号宽度是OOK两倍 
% 动态生成眼图的标题
dynamic_title = sprintf('Eye Diagram after DNN (PAM4 - %d bit)', quant);
title(dynamic_title);

% ---------------- 自动保存眼图 ----------------
h_eye = gcf; % 获取当前图形句柄 (Get Current Figure)

% 构造文件名: Eye_PRBS31_test_CNN_-15dBm.png
eye_filename = sprintf('Eye_%s_%s_%s_%ddBm.png', ...
    prbs_base_name, run_mode, model_type, received_optical_power);

full_eye_path = fullfile(image_save_path, eye_filename);

% 保存为高清 PNG
saveas(h_eye, full_eye_path);
fprintf('眼图已保存: %s\n', eye_filename);
% ----------------------------------------------------

% --- 自动推断 seq_len (delay) ---
total_sym_tx = length(original_pam_test_data); % 发射的总PAM符号数
total_sym_rx = length(received_eq);            % 接收的均衡后PAM符号数
lost_sym = total_sym_tx - total_sym_rx;

if lost_sym < 0 || mod(lost_sym, 2) ~= 0
    error('接收到的符号长度与发射长度不匹配，请检查是否使用了正确的 dnn_output.mat 文件。');
end
delay = lost_sym / 2; % 这就是PyTorch中的 seq_len
fprintf('自动推断的 seq_len (delay) 为: %s\n', num2str(delay));

% --- 检查延迟与PCM帧是否对齐 ---
delay_bits = delay * Mm; % Mm = log2(MM)
% if mod(delay_bits, quant) ~= 0
%     error(['致命错误：比特延迟 (', num2str(delay_bits), ') 不是 quant (', num2str(quant), ') 的整数倍。无法进行PCM帧同步。']);
% end
fprintf('当前比特延迟: %d (Quant=%d, 余数=%d)\n', delay_bits, quant, mod(delay_bits, quant));

%% ====================== PAM4 判决与解码 =========================
% --- PAM4 多门限判决 ---
% DNN输出 received_eq 的范围是 [-1, 1]
% 对应的理想电平是 [-1, -1/3, +1/3, +1]
% 判决门限是 -2/3, 0, 2/3
PAM_re_gray_symbols = zeros(1, length(received_eq)); % 预分配内存

PAM_re_gray_symbols(received_eq < -2/3) = 0;
PAM_re_gray_symbols((received_eq >= -2/3) & (received_eq < 0)) = 1;
PAM_re_gray_symbols((received_eq >= 0) & (received_eq < 2/3)) = 2; 
PAM_re_gray_symbols(received_eq >= 2/3) = 3;

% =========================================================================
% 自动寻找最佳对齐位置 (Lag Detection)
% =========================================================================
% 说明：由于 Rx_Part1 的帧同步可能存在抖动 (Jitter)，这里不能简单地使用固定 delay 截取。
% 我们使用互相关 (xcorr) 在接收符号流和理想标签之间寻找最佳偏移量 (Lag)。
% 扩大搜索范围以应对 Rx_Part1 回归可能带来的数据截断
search_window = min(length(original_pam_test_data), length(PAM_re_gray_symbols) + 50000);
[acor, lag_val] = xcorr(PAM_re_gray_symbols, original_pam_test_data(1:search_window));
[~, I_max] = max(abs(acor));
best_lag = lag_val(I_max);
fprintf('>>> SER计算对齐: 检测到最佳 Lag = %d\n', best_lag);

% 根据 best_lag 对齐数据
if best_lag > 0
    % Rx 滞后 (Lag > 0): 接收到的数据整体晚了，需要切掉 Rx 头部
    idx_rx_start = 1 + best_lag;
    idx_tx_start = 1;
else
    % Rx 超前 (Lag < 0): 接收到的数据整体早了 (包含了多余的前导?)，需要切掉 Tx 头部
    idx_rx_start = 1;
    idx_tx_start = 1 + abs(best_lag);
end

% 计算有效公共长度
len_avail_rx = length(PAM_re_gray_symbols) - idx_rx_start + 1;
len_avail_tx = length(original_pam_test_data) - idx_tx_start + 1;
len_common = min(len_avail_rx, len_avail_tx);

% 截取对齐后的序列 (用于 SER 计算和后续解码)
PAM_re_gray_symbols_aligned = PAM_re_gray_symbols(idx_rx_start : idx_rx_start + len_common - 1);
ref_data_aligned = original_pam_test_data(idx_tx_start : idx_tx_start + len_common - 1);

disp(['对齐后用于处理的符号长度: ', num2str(length(ref_data_aligned))]);

% --- SER 计算 ---
[SERnum,SERratio] = symerr(ref_data_aligned, PAM_re_gray_symbols_aligned);
% disp('--- PAM4 SER ---');
% disp(sprintf('SERratio =       %g', SERratio));
% --- 格雷码解码 (逆向映射) ---
% de_PAM1_gray 是恢复的格雷码符号序列 [0, 1, 3, 2, ...]
% 使用对齐后的符号流
de_PAM1_gray = PAM_re_gray_symbols_aligned'; 
% de_PAM1_decimal 是解码后的十进制序列 [0, 1, 2, 3, ...]
de_PAM1_decimal = gray2bin(de_PAM1_gray, 'pam', MM);

% --- 十进制 to 比特 ---
% 将每个PAM4十进制符号转换为2个比特
Bin1_matrix = de2bi(de_PAM1_decimal, Mm, 'left-msb');
% 将比特矩阵序列化为单个比特流
Bin1_stream = reshape(Bin1_matrix', [], 1);

%% ====== PCM帧对齐与后续处理 ======
% --- 帧对齐 (基于 symbol lag 的计算) ---
% 我们必须根据 PAM 符号的对齐情况，来推算 PCM 数据的起始填充位置

% 1. 计算比特流相对于原始数据的相位
% ref_data_aligned 是从 original_pam_test_data 的第 idx_tx_start 个符号开始的
% 对应的原始比特索引是 (idx_tx_start - 1) * Mm + 1

bit_start_index = (idx_tx_start - 1) * Mm; % 这是前面跳过的比特数


% 2. 确保 PCM 帧边界对齐
% 我们需要 Bin1_stream 的第一个比特对应一个 PCM 采样的 MSB (第1位)
% 如果 bit_start_index 不是 quant (8) 的倍数，说明当前 Bin1_stream 的开头不是完整的 PCM 采样
bits_to_skip = mod(bit_start_index, quant);

if bits_to_skip > 0
    % 丢弃开头不完整的比特，对齐到下一个 PCM 采样边界
    Bin1_stream = Bin1_stream(1 + (quant - bits_to_skip) : end);
    % 更新跳过的总比特数
    bit_start_index = bit_start_index + (quant - bits_to_skip);
end

% 计算 PCM 偏移量 (丢弃了多少个完整的 PCM 采样)
% 计算在原始 OFDM 序列中的起始采样点偏移量
% 这是为了后面填入 DA_code 时能对上号
pcm_start_offset = bit_start_index / quant; 

% 3. 截断为整数个 PCM 采样
num_valid_pcm = floor(length(Bin1_stream) / quant);
Bin1_stream = Bin1_stream(1 : num_valid_pcm * quant);
samplenumber_rx = num_valid_pcm;

% 4. PCM 解码
quant_Bin1 = reshape(Bin1_stream, quant, samplenumber_rx)';
quant_Dec1 = bi2de(quant_Bin1, 'left-msb') + 1;
quant_OFDM_re = zeros(samplenumber_rx, 1);
for ii=1:samplenumber_rx
  jj=quant_Dec1(ii,1);
  quant_OFDM_re(ii,1)=codebook(1,jj);   % 恢复的量化码
end

%% ========================================================================
% [核心模块] PCM 质量诊断、自动对齐与波形重建
% =========================================================================

% --- 1. PAM4 符号质量检查 ---
% ref_data_aligned 是理想符号, PAM_re_gray_symbols_aligned 是判决后符号
% 我们计算判决前的软信息 (received_eq) 的 MSE
if best_lag > 0
    rec_aligned_soft = received_eq(1 + best_lag : end);
    ref_aligned_soft = original_pam_test_data;
else
    rec_aligned_soft = received_eq;
    ref_aligned_soft = original_pam_test_data(1 + abs(best_lag) : end);
end
len_soft = min(length(rec_aligned_soft), length(ref_aligned_soft));
rec_soft_cut = rec_aligned_soft(1:len_soft);
ref_soft_cut = ref_aligned_soft(1:len_soft);

% 将理想符号映射回电平 [-1, -1/3, 1/3, 1] 以便比较
% 假设 original_pam_test_data 是 [0, 1, 2, 3] 格式
ref_levels = zeros(size(ref_soft_cut));
ref_levels(ref_soft_cut == 0) = -1;
ref_levels(ref_soft_cut == 1) = -1/3;
ref_levels(ref_soft_cut == 2) = 1/3;
ref_levels(ref_soft_cut == 3) = 1;

% 计算符号级 MSE
pam_mse = mean((rec_soft_cut - ref_levels).^2);
fprintf('\n--- 信号质量诊断 ---\n');
fprintf('>> PAM4 符号级 MSE: %.6f (越小越好)\n', pam_mse);
fprintf('>> PAM4 SER: %.4e (参考值)\n', SERratio);

if SERratio < 1e-2 && pam_mse > 0.1
    warning('警告：SER 很低但 MSE 很高，说明符号虽然判决对了，但离中心很远（噪声大）。');
end

% --- 2. PCM 序列对齐检查 ---
% (1) 重建 Tx 端标准 PCM 序列 (Ground Truth)
% 这一步必须保证绝对正确，直接利用 Tx 参数生成
bp_OFDM_Tx_Diagnostic = zeros(1, samplenumber);
for m=1:samplenumber
    bp_OFDM_Tx_Diagnostic(1,m) = OFDM(1,(m-1)*R+1); 
end

% (2) 重新计算 partition (防止变量丢失)
V_max_diag = max(abs(bp_OFDM_Tx_Diagnostic));
V_int_diag = fix(V_max_diag) + 1;
quantDelta_diag = 2 * V_int_diag / (2^quant);
partition_diag = (-V_int_diag : quantDelta_diag : V_int_diag); 
% 注意：quantiz 返回的 codebook索引 是从0开始的，我们需要加1来匹配 codebook 数组索引

% (3) 重新量化得到 Tx 的理想电压值 (Ground Truth)
[~, Tx_PCM_GT] = quantiz(bp_OFDM_Tx_Diagnostic, partition_diag, codebook);

% (4) 获取 Rx PCM 序列
Rx_PCM_Recovered = quant_OFDM_re.'; % 转置为行向量

% (5) 执行互相关扫描 (寻找是否还有残留位移)
% 我们截取一段数据进行精确比对
check_len = min([10000, length(Tx_PCM_GT), length(Rx_PCM_Recovered)]);
tx_segment = Tx_PCM_GT(1:check_len);
rx_segment = Rx_PCM_Recovered(1:check_len);

% 计算互相关
[xc, lags] = xcorr(rx_segment - mean(rx_segment), tx_segment - mean(tx_segment), 100, 'coeff');
[max_corr, max_idx] = max(abs(xc));
detected_pcm_lag = lags(max_idx);

fprintf('>> PCM 序列相关性分析:\n');
fprintf('   最大相关系数: %.4f (应接近 1.0)\n', max_corr);
fprintf('   检测到的残留偏移 (Lag): %d (应为 0)\n', detected_pcm_lag);

% --- 3. [自动修正] 应用 PCM 对齐并计算最终 SQNR ---
if detected_pcm_lag ~= 0
    fprintf('   >>> 正在自动修正 PCM 偏移 (Lag = %d)...\n', detected_pcm_lag);
    if detected_pcm_lag < 0
        n_pad = abs(detected_pcm_lag);
        n_pad = floor(n_pad); % 确保是整数
        padding = zeros(n_pad, 1); 
        % 头部补零，尾部截断，保持 quant_OFDM_re (列向量) 长度不变
        quant_OFDM_re = [padding; quant_OFDM_re(1:end-n_pad)];
    else % detected_pcm_lag > 0
        n_cut = abs(detected_pcm_lag);
        n_cut = floor(n_cut); % 确保是整数
        padding = zeros(n_cut, 1);
        % 头部截断，尾部补零
        quant_OFDM_re = [quant_OFDM_re(n_cut+1:end); padding];
    end
end





% % =========================================================================
% % 理想 FEC 模拟 (Ideal FEC Emulation)
% % =========================================================================
% % 方法：利用 Ground Truth 识别并纠正那些大幅度的 MSB 误码 (Impulse Noise)，
% % 保留小幅度的量化噪声和 DNN 拟合误差。
% % =========================================================================
% % 1. 确保长度对齐以进行比较
% len_fix = min(length(quant_OFDM_re), length(Tx_PCM_GT));
% rx_temp = quant_OFDM_re(1:len_fix);
% tx_temp = Tx_PCM_GT(1:len_fix).'; % 确保转置为列向量
% 
% % 2. 设定误码判定门限 (Threshold)
% % 理想 FEC (Ideal FEC) 的原理是基于"比特"纠错，而非"幅度"。
% % 只要存在比特翻转，无论由此导致的电压误差是大是小，FEC 都会将其纠正。
% % 由于 Rx 和 Tx 都是离散的量化电平，任何不相等 (Diff > 0) 都意味着发生了误码。
% % 因此，我们使用极小的门限 (仅容忍浮点计算误差)，以模拟"纠正所有比特错误"的效果。
% err_threshold = 1e-4;
% 
% % 3. 检测误码位置
% diff_vec = abs(rx_temp - tx_temp);
% is_bit_error = diff_vec > err_threshold;
% 
% % 4. 执行纠错 (Genie-Aided Correction)
% rx_temp(is_bit_error) = tx_temp(is_bit_error);
% 
% % 5. 写回数据
% quant_OFDM_re(1:len_fix) = rx_temp;
% 
% fprintf('       [FEC] 已纠正误码点数: %d (%.2f%%)\n', sum(is_bit_error), 100*mean(is_bit_error));











% 计算最终修正后的 SQNR (这是最准确的值)
Rx_PCM_Final = quant_OFDM_re; % 列向量
len_calc = min(length(Tx_PCM_GT), length(Rx_PCM_Final));
sig_power = mean(abs(Rx_PCM_Final(1:len_calc)).^2);
% 注意维度匹配: Tx_PCM_GT是行, Rx_PCM_Final是列, 需要转置 Tx
noise_power = mean(abs(Tx_PCM_GT(1:len_calc).' - Rx_PCM_Final(1:len_calc)).^2);
PCM_SQNR_Fixed = 10*log10(sig_power / noise_power);
PCM_SQNR = PCM_SQNR_Fixed; % 更新全局变量

fprintf('   >>> 最终 PCM SQNR: %.2f dB\n', PCM_SQNR);

% --- 4. [可视化验证] 绘图验证 PCM 对齐效果 ---
figure('Name', 'Verification: Aligned PCM Waveform');
% 截取前200个点进行清晰展示
plot_len = min(200, len_calc);
plot(Tx_PCM_GT(1:plot_len), 'k-', 'LineWidth', 1.5); hold on;
plot(Rx_PCM_Final(1:plot_len), 'g--', 'LineWidth', 1.5);
legend('Tx Ground Truth', 'Rx Fixed (Aligned)');
title(sprintf('Aligned PCM Waveform (SQNR: %.2f dB)', PCM_SQNR));
grid on; xlabel('Sample Index'); ylabel('Voltage');
% 保存这个关键图片
saveas(gcf, fullfile(image_save_path, sprintf('PCM_Aligned_%s_%s_%dbit.png', prbs_base_name, run_mode, quant)));

% --- 5. [波形重建] 使用固定 R=4 重建 5GHz 信号 ---
% =========================================================================
% 核心逻辑：
% 1. 使用 Step 3 对齐后的 PCM 数据 (quant_OFDM_re)。
% 2. 强制按照 R=4 (即 200ps 间隔) 将其填入 DA_code。
% 3. 这将生成一个标准的 5GHz 信号，能够顺利通过 BandPF。
% =========================================================================
fprintf('   >>> 正在重建 5GHz 标准波形 (Fixed R=%d)...\n', R);
DA_code = zeros(length(OFDM), 1);

% 使用对齐后的 PCM 数据进行填充
% 注意：quant_OFDM_re 已经是经修正后的列向量
num_samples_to_fill = min(length(quant_OFDM_re), floor(length(DA_code)/R));

for i = 1 : num_samples_to_fill
    % 计算在 OFDM 帧中的位置 (固定间隔 R)
    idx_in_ofdm = (i - 1) * R + 1;
    
    % 填入 PCM 值
    DA_code(idx_in_ofdm) = quant_OFDM_re(i);
end

% 带通滤波 (提取重建后的 5GHz 信号)
% 注意：因为 DA_code 是稀疏的 (插0)，所以会有高频镜像，BandPF 会提取出 5GHz 处的那个镜像
re_OFDM = BandPF(DA_code', Lf, Uf, Fs);

fprintf('   >>> 波形重建完成。进入 OFDM 解调...\n');

%% ====================== OFDM 解调与性能评估 ======================
% 1. 带通滤波 (去除重采样和量化引入的带外噪声)
%    re_OFDM_resampled 已经是 Fs 采样率了，直接滤波
% % re_OFDM = BandPF(re_OFDM_resampled', Lf, Uf, Fs);
% 使用重建好的 DA_code 作为输入
% 注意：DA_code 是列向量，BandPF 需要行向量，所以要加转置 '
re_OFDM = BandPF(DA_code', Lf, Uf, Fs);

% 2. 丢弃由于 PCM 对齐和重采样带来的头部不稳定数据
%    pcm_start_offset 导致我们丢失了前面的 PCM 采样
%    重采样滤波器也有群延迟
%    简单起见，我们丢弃前 N 个点，或者让后面的循环自动找 (OFDM 需要同步吗？)
%    注意：这里假设 re_OFDM 的开头已经大体对齐了（因为 quant_OFDM_re 是从流的开头解出来的）
%    微小的符号内偏移 (Phase Rotation) 会由 MatchFilter 和 EVM 计算中的星座图旋转消除。

% ---------------  S2P ------------------------------
Rx_signal=zeros(SymPerCar,(1+CP)*N_IFFT*Rate);
max_len = length(re_OFDM);

for ii=1:SymPerCar
  start_idx = (ii-1)*((1+CP)*N_IFFT)*Rate + 1;
  end_idx = ii*((1+CP)*N_IFFT)*Rate;
  
  if end_idx <= max_len
      Rx_signal(ii,:) = re_OFDM(1, start_idx:end_idx);
  else
      % 如果数据不够，补零 (防止报错)
      len_needed = end_idx - start_idx + 1;
      avail_len = max_len - start_idx + 1;
      if avail_len > 0
          Rx_signal(ii,:) = [re_OFDM(1, start_idx:end), zeros(1, len_needed - avail_len)];
      else
          Rx_signal(ii,:) = zeros(1, len_needed);
      end
  end
end

%  -------------  carrier demod ----------------------
reCarrier_OFDM=zeros(SymPerCar,(1+CP)*N_IFFT*Rate);
for ii = 1:SymPerCar
  reCarrier_OFDM(ii,:) = Rx_signal(ii,:).*cos(2*pi*fc*symLen*dt);
end

% --------- down sample (Match Filter) ---------------
match_OFDM=zeros(SymPerCar,N_IFFT*(1+CP));
delay_OFDM=zeros(SymPerCar,N_IFFT*(1+CP)+2*Delay);
for ii=1:SymPerCar
  % 匹配滤波
  delay_OFDM(ii,:)=MatchFilter(reCarrier_OFDM(ii,:),Fs,Rs,Rolloff,Delay);
  % 同发射端，移除滤波器带来的群延迟
  match_OFDM(ii,:)=delay_OFDM(ii,Delay+1:end-Delay);
end

% -------  Remove the loop prefix ----------------
minusCP=match_OFDM(:,CP*N_IFFT+1:(1+CP)*N_IFFT);

% ------- FFT --------------
% OFDM decode FFT
FFT=fft(minusCP,N_IFFT,2);
% Divide the zeros added by the IFFT/FFT transform and pick out the mapped subcarriers
re_QAM = FFT(:,Subcarrier);
% p2s
re_QAM=reshape(re_QAM.',1,N_Sc*SymPerCar);

%% ====================Normalized Received Signal==============
% Normalized power of transmitted signal
sumP_base_QAM=0;
for ii=1:length(base_QAM)
  P_base_QAM=(real(base_QAM(1,ii)))^2+(imag(base_QAM(1,ii)))^2;
  sumP_base_QAM=sumP_base_QAM+P_base_QAM;
end
averP_base_QAM=sqrt(sumP_base_QAM/M);

% 归一化接收信号的功率
sumP_re_QAM=0;
for ii=1:length(re_QAM)
  P_re_QAM=(real(re_QAM(1,ii)))^2+(imag(re_QAM(1,ii)))^2;
  sumP_re_QAM=sumP_re_QAM+P_re_QAM;
end
averP_re_QAM=sqrt(sumP_re_QAM/M);
RR=averP_base_QAM/averP_re_QAM;
re_QAM1=re_QAM*RR;

% --- 绘制星座图 ---
figure;
plot(real(re_QAM1), imag(re_QAM1), '.', 'MarkerSize', 6);
hold on;
plot(real(base_QAM), imag(base_QAM), 'r+', 'MarkerSize', 10, 'LineWidth', 1.5);
grid on;
axis([-4 4 -4 4]);
xlabel('In-Phase');
ylabel('Quadrature');
title(sprintf('Constellation (Quant=%d)', quant));
legend('Received Symbols', 'Ideal Constellation');

% ---------------- [新增] 自动保存星座图 ----------------
h_const = gcf; % 获取当前图形句柄

% 构造文件名: Constellation_PRBS31_test_CNN_-15dBm.png
const_filename = sprintf('Constellation_%s_%s_%s_%ddBm.png', ...
    prbs_base_name, run_mode, model_type, received_optical_power);

full_const_path = fullfile(image_save_path, const_filename);

% 保存为高清 PNG
saveas(h_const, full_const_path);
fprintf('星座图已保存: %s\n', const_filename);
% -------------------------------------------------------

% --- EVM & BER ---
evm=comm.EVM;
rmsEVM=step(evm,base_QAM.',re_QAM1.');
% disp('--- rmsEVM ---');
% disp(sprintf('rmsEVM =       %f', rmsEVM));

% OFDM BER
re_base=qamdemod(re_QAM1,M,'gray');
re_base1=de2bi(re_base','left-msb');
re_base2=reshape(re_base1',1,length(Txdata));
% 不要用波浪号 ~, 把误码个数接出来
[BERnum, BERratio] = biterr(Txdata, re_base2); 

%% === FINAL REPORT 输出 & 自动保存 ===
% 1. 构造文件名: Report_PRBS31_test_CNN_-15dBm.txt
report_filename = sprintf('Report_%s_%s_%s_%ddBm.txt', ...
    prbs_base_name, run_mode, model_type, received_optical_power);
full_report_path = fullfile(report_save_path, report_filename);

% 2. 打开文件 (权限 'w' 表示覆盖写入，如果想追加用 'a')
fid = fopen(full_report_path, 'w');

% 3. 构造要输出的内容字符串 (方便同时打印到屏幕和文件)
% 使用 cell 数组存储每一行，方便循环处理
report_lines = {};
report_lines{end+1} = '========================================';
report_lines{end+1} = '        FINAL PERFORMANCE REPORT        ';
report_lines{end+1} = '========================================';
report_lines{end+1} = sprintf('Data Source   : %s (%s) %d dBm', prbs_base_name, run_mode, received_optical_power);
report_lines{end+1} = sprintf('Quantization  : %d bit', quant);
report_lines{end+1} = sprintf('Method        : Fractional Resampling');
report_lines{end+1} = sprintf('Model Type    : %s', model_type); 
report_lines{end+1} = sprintf('SER (PAM4)    : %.4e', SERratio); 
report_lines{end+1} = sprintf('PCM SQNR      : %.4f dB', PCM_SQNR);
report_lines{end+1} = sprintf('rms EVM       : %.4f %%', rmsEVM);
report_lines{end+1} = sprintf('BER (16-QAM)  : %.4e', BERratio);
report_lines{end+1} = sprintf('Total Errors  : %d / %d bits', BERnum, length(Txdata));
report_lines{end+1} = '========================================';
report_lines{end+1} = sprintf('Date          : %s', datestr(now)); 

% 4. 执行输出
for i = 1:length(report_lines)
    % 打印到 MATLAB 命令行 (屏幕)
    disp(report_lines{i});
    
    % 打印到 .txt 文件 (换行符 \r\n 适配 Windows 记事本)
    fprintf(fid, '%s\r\n', report_lines{i});
end

% 5. 关闭文件
fclose(fid);
fprintf('性能报告已自动保存到: %s\n', full_report_path);
disp('脚本运行结束。');
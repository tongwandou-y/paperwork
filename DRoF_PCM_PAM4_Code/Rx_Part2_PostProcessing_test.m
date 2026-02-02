% =========================================================================
% DRoF - Rx_Part2_PostProcessing_test.m
% 目的：加载DNN均衡后的 (PRBS31) 信号，并进行性能评估
%
% 流程:
% 1. 加载 PRBS31_test.mat (参数)
% 2. 加载 Data_For_NN_PRBS31_test.mat (理想标签)
% 3. 加载 NN_Output_test.mat (均衡结果)
% 4. 执行 PAM4 判决, SER, PCM 解码, OFDM 解调, 最终 BER 计算
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
quant = 8; % 手动修改为量化比特数 (必须与发射端一致！)

% 模型类型选择: 'DNN' 或 'CNN'
% 必须与 Python configs.py 中的 config.model_type 保持一致！
model_type = 'DNN';

root_base_dir = 'D:\paperwork\Experiment_Data\10Gsyms_20km';

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
% 构造参数文件名 (例如: DRoF_PCM_Parameters_PRBS31_test.mat)
param_filename = sprintf('DRoF_PCM_Parameters_%s_%s.mat', prbs_base_name, run_mode);
% 组合完整路径
param_full_path = fullfile(param_load_path, param_filename);

disp(['加载参数: ', param_full_path]);

% 检查文件是否存在
if ~exist(param_full_path, 'file')
    error('错误: 找不到参数文件 \n%s \n请检查发射端脚本是否已运行并保存到正确位置。', param_full_path);
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
quant = DRoF_PCM_Parameters.PCM_quant;
codebook = DRoF_PCM_Parameters.PCM_codebook;
MM  = DRoF_PCM_Parameters.PAM_MM;
Mm  = DRoF_PCM_Parameters.PAM_Mm;
% PAM_code = DRoF_PCM_Parameters.PAM_code; % 我们将从 test label 文件加载
R_AWG  = DRoF_PCM_Parameters.RRC_R_AWG;
sps  = DRoF_PCM_Parameters.RRC_sps;
Fs_AWG = DRoF_PCM_Parameters.Fs_AWG;
beta = DRoF_PCM_Parameters.RRC_beta;
span = DRoF_PCM_Parameters.RRC_N_span;

%% === 3. 加载理想标签和DNN均衡结果 (TEST) ===
% -------------------------------------------------------------------------
% 1. 加载理想测试标签 (来自 MATLAB Part 1)
% -------------------------------------------------------------------------
% 格式例如: Data_For_NN_PRBS23_test_-18.mat
label_filename = sprintf('Data_For_NN_%s_%s_%d.mat', prbs_base_name, run_mode, received_optical_power);
full_label_path = fullfile(mat_input_path, label_filename);

disp(['加载理想标签: ', full_label_path]);
if ~exist(full_label_path, 'file')
    error(['找不到文件: ', full_label_path, ' 请检查路径或是否已运行 Part 1。']);
end
% 仅加载 'original_pam_test_data' (格式 [0,1,2,3])
load(full_label_path, 'original_pam_test_data');

% -------------------------------------------------------------------------
% 2. 加载DNN均衡结果 (来自 Python)
% -------------------------------------------------------------------------
% 按照您指定的格式构造文件名: NN_Output_test_DNN_-27.mat
% 对应格式: NN_Output_{run_mode}_{model_type}_{power}.mat
dnn_output_filename = sprintf('NN_Output_%s_%s_%d.mat', run_mode, model_type, received_optical_power);
dnn_output_full_path = fullfile(mat_output_path, dnn_output_filename);

disp(['加载DNN均衡结果: ', dnn_output_full_path]);
if ~exist(dnn_output_full_path, 'file')
    error(['找不到文件: ', dnn_output_full_path, ' 请检查Python脚本是否已运行并保存为带功率后缀的文件名。']);
end
disp(['加载DNN均衡结果: ', dnn_output_filename]);
load(dnn_output_full_path); % 加载一个名为 received_eq 的变量

%% ==================== 信号格式化及参数自动推断 ======================
% % 确保数据是行向量
% if size(received_eq, 1) > 1
%     received_eq = received_eq.';
% end
% 
% disp('正在绘制 NN 均衡后眼图...');
% 
% % 显示均衡后的眼图
% % h_eye = eyediagram(received_eq(1, 1:min(2000, length(received_eq))), 2*sps/Mm); % PAM4眼图符号宽度是OOK两倍
% h_eye = eyediagram(received_eq(1, 1:min(2000, length(received_eq))), 2); % PAM4眼图符号宽度是OOK两倍

%% ==================== 信号格式化及参数自动推断（二） ======================
% 确保数据是行向量
if size(received_eq, 1) > 1
    received_eq = received_eq.';
end

% -------------------------------------------------------------------------
% [核心修改] 重建平滑眼图 (Reconstruct Smooth Eye Diagram)
% 原理：NN输出是1 SPS的离散点。为了看清眼图的张开和过渡轨迹，
% 我们需要进行上采样(Upsampling)和成型滤波(Pulse Shaping)。
% 优化策略：仅截取前 N 个符号进行插值和绘图，避免卡死
% -------------------------------------------------------------------------
disp('正在重建并绘制 NN 平滑眼图...');

% 1. 设置插值参数
upsample_factor = 16;  % 上采样倍数 (越高越平滑，16通常足够模拟模拟波形)
filter_span = 6;       % 滤波器跨度 (符号数)
% 使用脚本前面加载的 Rolloff 参数，如果没有加载则默认 0.2
% if exist('Rolloff', 'var')
%     beta_val = Rolloff;
% else
beta_val = 0.5; 
% end

% 2. 设计成型滤波器 (根升余弦 RRC)
% 这模拟了信号在DAC之后的模拟形态
rrcFilter = rcosdesign(beta_val, filter_span, upsample_factor);

% 3. 【关键优化】截取部分数据进行绘图 (第10000到15000)
idx_start = 10000;
idx_end   = 20000;
% 截取该区间的信号
sig_subset = received_eq(idx_start : idx_end);

% 4. 对截取的数据进行上采样
sig_upsampled = upsample(sig_subset, upsample_factor);
% 5. 卷积滤波 (填补零值，形成平滑曲线)
% 注意：这只是为了画图好看，不影响 BER 计算！BER 计算依然使用原始的 received_eq
sig_smooth = conv(sig_upsampled, rrcFilter, 'same');

% 5. 处理滤波带来的幅度衰减和群延迟
% 简单的归一化以匹配 [-1, 1] 范围，方便画参考线
sig_smooth = sig_smooth / max(abs(sig_smooth)); 

% 6. 绘制眼图
% 注意：这里输入的是 sig_smooth (插值后的短数据)
% 这里的 '2 * upsample_factor' 意思是：每条轨迹显示 2 个符号周期。
% 因为现在每个符号包含了 upsample_factor 个采样点。
h_eye = eyediagram(sig_smooth, 2 * upsample_factor);


% ==================================================================


% ---------------- 自动保存眼图 ----------------
% 动态生成眼图的标题
dynamic_title = sprintf('Eye Diagram after DNN (PAM4 - %s TEST %d dBm)', prbs_base_name, received_optical_power);
% title(dynamic_title);
set(h_eye, 'Name', dynamic_title); % 设置窗口名称

h_eye = gcf; % 获取当前图形句柄 (Get Current Figure)

% 构造文件名: Eye_PRBS31_test_DNN_-15dBm.png
eye_filename = sprintf('Eye_%s_%s_%s_%ddBm.png', ...
    prbs_base_name, run_mode, model_type, received_optical_power);

full_eye_path = fullfile(image_save_path, eye_filename);

% 保存为高清 PNG
saveas(h_eye, full_eye_path);
fprintf('眼图已保存: %s\n', eye_filename);

% --- 自动推断 seq_len (delay) ---
% 【注意】计算 delay 必须使用原始的全量 received_eq，不能用截取后的 sig_subset！
total_sym_tx = length(original_pam_test_data); 
total_sym_rx = length(received_eq);            
lost_sym = total_sym_tx - total_sym_rx;

if lost_sym < 0 || mod(lost_sym, 2) ~= 0
    error('接收到的符号长度与发射长度不匹配，请检查是否使用了正确的 dnn_output.mat 文件。');
end
delay = lost_sym / 2; % 这就是PyTorch中的 seq_len
disp(['自动推断的 seq_len (delay) 为: ', num2str(delay)]);

% --- 检查延迟与PCM帧是否对齐 ---
delay_bits = delay * Mm; % Mm = log2(MM)
if mod(delay_bits, quant) ~= 0
    error(['致命错误：比特延迟 (', num2str(delay_bits), ') 不是 quant (', num2str(quant), ') 的整数倍。无法进行PCM帧同步。']);
end

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

max_lag_search = 100; % 设置搜索范围 (+/- 100 符号)
% 截取一部分数据进行互相关，提高计算效率
len_corr = min(length(PAM_re_gray_symbols), length(original_pam_test_data));

[acor, lag_val] = xcorr(PAM_re_gray_symbols(1:len_corr), original_pam_test_data(1:len_corr), max_lag_search);
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

disp(['均衡后判决符号长度 (原始): ', num2str(length(PAM_re_gray_symbols))]);
disp(['对齐后用于处理的符号长度: ', num2str(length(ref_data_aligned))]);

% --- SER 计算 (在格雷码符号上) ---
[SERnum,SERratio] = symerr(ref_data_aligned, PAM_re_gray_symbols_aligned);
% disp('--- PAM4 SER ---');
% disp(sprintf('SERratio =       %g', SERratio));



% =========================================================================
% PAM Link BER 计算
% =========================================================================
% 原理：利用已经对齐的符号流 (ref_data_aligned 和 PAM_re_gray_symbols_aligned)
% 将它们分别通过 gray2bin 和 de2bi 还原成比特流，直接计算物理链路的 BER。

% 1. 将对齐后的【理想发送符号】转回比特
% ref_data_aligned 是 0,1,2,3 的电平值 (遵循格雷码)
Tx_Dec_Aligned = gray2bin(ref_data_aligned, 'pam', MM);  % 格雷逆映射 -> 十进制
Tx_Bits_Mat = de2bi(Tx_Dec_Aligned, Mm, 'left-msb');     % 十进制 -> 二进制矩阵
Tx_Bits_Link = reshape(Tx_Bits_Mat', [], 1);             % 拉直成比特流

% 2. 将对齐后的【实际接收符号】转回比特
Rx_Dec_Aligned = gray2bin(PAM_re_gray_symbols_aligned', 'pam', MM); % 注意转置，确保是列向量
Rx_Bits_Mat = de2bi(Rx_Dec_Aligned, Mm, 'left-msb');
Rx_Bits_Link = reshape(Rx_Bits_Mat', [], 1);

% 3. 计算 PAM 链路层的 BER
[BER_PAM_Num, BER_PAM_Ratio] = biterr(Tx_Bits_Link, Rx_Bits_Link);

fprintf('>>> PAM Link BER: %.4e (Errors: %d/%d)\n', ...
    BER_PAM_Ratio, BER_PAM_Num, length(Tx_Bits_Link));
% =========================================================================





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

%% ====================== PCM帧对齐与后续处理 ======================
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

% 3. 计算在原始 OFDM 序列中的起始采样点偏移量
% 这是为了后面填入 DA_code 时能对上号
pcm_start_offset = bit_start_index / quant; 

% 确保比特流长度是 quant 的整数倍 (丢弃末尾不足一个采样的比特)
num_valid_pcm = floor(length(Bin1_stream) / quant);
Bin1_stream = Bin1_stream(1 : num_valid_pcm * quant);
samplenumber_rx = num_valid_pcm;

% --- PCM 解码 ---
quant_Bin1 = reshape(Bin1_stream, quant, samplenumber_rx)';
quant_Dec1 = bi2de(quant_Bin1, 'left-msb') + 1;
quant_OFDM_re = zeros(samplenumber_rx, 1);
for ii=1:samplenumber_rx
    jj=quant_Dec1(ii,1);
    quant_OFDM_re(ii,1)=codebook(1,jj);     % 恢复的量化码
end

% --- SQNR 计算 (对齐后) ---
% 1. 加载原始的 PCM 符号 (全部)
% 注意：original_pam_test_data 来源于 Txdata -> PCM -> PAM
% 这里我们直接重新从 OFDM 采样一下，确保是原始全集
bp_OFDM_Tx_full = zeros(1, samplenumber);
for m=1:samplenumber
    bp_OFDM_Tx_full(1,m) = OFDM(1,(m-1)*R+1);   
end

% 2. 截取对应的原始片段进行比较
% 我们知道 quant_OFDM_re 对应的原始位置是从 pcm_start_offset + 1 开始的
idx_start_sqnr = pcm_start_offset + 1;
idx_end_sqnr = pcm_start_offset + samplenumber_rx;

% 边界检查
if idx_end_sqnr <= length(bp_OFDM_Tx_full)
    bp_OFDM_Tx_aligned = bp_OFDM_Tx_full(idx_start_sqnr : idx_end_sqnr)';
    quant_OFDM_re_temp = quant_OFDM_re; % 已经是列向量
    
    % 4. 在对齐的数据上计算SQNR
    Power_QuantSignal = mean(abs(quant_OFDM_re_temp).^2);
    QuantNoise = bp_OFDM_Tx_aligned - quant_OFDM_re_temp;
    Power_QuantNoise = mean(abs(QuantNoise).^2);
    PCM_SQNR = 10*log10(Power_QuantSignal/Power_QuantNoise);
else
    warning('SQNR 计算越界，跳过');
    PCM_SQNR = 0;
end
% disp('--- PCM SQNR ---');
% disp(sprintf('PCM_SQNR =    %f', PCM_SQNR));

% --- 上采样与OFDM帧重建 ---
% 我们必须将恢复的符号放置在正确的时间偏移上
DA_code=zeros(length(OFDM),1); % 创建一个全零的、原始长度的向量

for i=1:samplenumber_rx
    % [修改] 使用计算出的 pcm_start_offset 进行填充
    % 原始位置: i + pcm_start_offset
    idx_in_ofdm = R * (i - 1 + pcm_start_offset) + 1;
    
    if idx_in_ofdm <= length(DA_code)
        DA_code(idx_in_ofdm, 1) = quant_OFDM_re(i, 1);
    end
end

%% ====================== OFDM 解调与性能评估 ======================
re_OFDM = BandPF(DA_code',Lf,Uf,Fs);
% ---------------   S2P ------------------------------
Rx_signal=zeros(SymPerCar,(1+CP)*N_IFFT*Rate);
for ii=1:SymPerCar
    Rx_signal(ii,:)=re_OFDM(1,(ii-1)*((1+CP)*N_IFFT)*Rate+1:ii*((1+CP)*N_IFFT)*Rate);
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
title('Constellation Diagram (TEST)');
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
report_lines{end+1} = sprintf('Data Source   	: %s (%s) %d dBm', prbs_base_name, run_mode, received_optical_power);
report_lines{end+1} = sprintf('Model Type    	: %s', model_type);

% --- 物理链路层指标 (PAM4) ---
report_lines{end+1} = '----------------------------------------';
report_lines{end+1} = sprintf('SER (PAM4)    	: %.4e', SERratio);       % 符号误码率
report_lines{end+1} = sprintf('BER (PAM4)       : %.4e', BER_PAM_Ratio); % 链路比特误码率
report_lines{end+1} = sprintf('PAM4 Total Errors: %d / %d bits', BER_PAM_Num, length(Tx_Bits_Link)); % 具体的错误个数 (你刚才加的这行)

% --- 中间层指标 (PCM & OFDM) ---
report_lines{end+1} = '----------------------------------------';
report_lines{end+1} = sprintf('PCM SQNR      	: %.4f dB', PCM_SQNR);
report_lines{end+1} = sprintf('rms EVM       	: %.4f %%', rmsEVM);

% --- 应用层指标 (16-QAM) ---
report_lines{end+1} = '----------------------------------------';
report_lines{end+1} = sprintf('BER (16-QAM) 	: %.4e', BERratio);
report_lines{end+1} = sprintf('Total Errors  	: %d / %d bits', BERnum, length(Txdata));

report_lines{end+1} = '========================================';
report_lines{end+1} = sprintf('Date          	: %s', datestr(now));


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
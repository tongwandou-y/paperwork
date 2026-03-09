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
received_optical_power = -15;     % 设置接收光功率 (对应文件名中的数字，如 -10, -14, -18 等)
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


%% === 4. 信号处理与同步 (全局处理 + 符号级直接对齐) ===
% ====================================================================
% 方法说明：
%   之前使用波形级互相关（重建FFE+RRC前导波形与received做xcorr）来帧同步，
%   但由于VPI光链路对波形的非线性变换，重建的模板与接收波形形状不匹配，
%   导致xcorr找到的是假峰 (frameLoc≈1)，帧同步彻底失败。
%
%   新方法（零标签泄漏）：
%   1. 对完整接收信号做低通滤波 + 基线漂移消除
%   2. 在每个采样相位上分别下采样，并用已知SYNC模板做互相关
%   3. 联合估计最优采样相位、同步起点和极性
%   4. 仅用SYNC模板进行质量验证，不使用PAM_code参与对齐
%
%   优势：符号级xcorr不依赖波形形状匹配，只要求接收符号与发射符号
%         存在线性（或近似线性）关系即可。PRBS派生的PAM4序列具有
%         尖锐的自相关峰，提供了极高的同步可靠性。
% ====================================================================
fs = Fs_AWG;
Baudrate_pam = fs / sps;
pam_data_symbols = length(PAM_code);

disp('============= 全局信号处理 + 联合同步 =============');

% ======================== 步骤1: 全信号低通滤波 ========================
disp('步骤1: 对完整接收信号低通滤波...');
stream_full = LowPF(received, -1*Baudrate_pam, Baudrate_pam, fs);
stream_full = real(stream_full);

% ======================== 步骤2: 基线漂移消除 ========================
disp('步骤2: 消除基线漂移...');
window_size = round(sps * 200);
stream_full = stream_full - movmean(stream_full, window_size);

% ======================== 步骤3: 联合定时恢复与帧同步 ========================
disp('步骤3: 执行联合定时恢复与帧同步 (多相位 + SYNC模板)...');

% 仅使用TX侧保存的同步模板（FFE后符号级参考），避免旧流程混用造成错位
if ~(isfield(DRoF_PCM_Parameters, 'SYNC_ref_after_ffe') && ~isempty(DRoF_PCM_Parameters.SYNC_ref_after_ffe))
    error(['当前参数文件缺少 SYNC_ref_after_ffe。请按以下顺序重新生成数据:\n' ...
           '  1) 运行 OFDM_PCM_PAM4tx_atuo.m 生成新参数和新TX波形\n' ...
           '  2) 在VPI中使用新TX波形重新仿真并导出CSV\n' ...
           '  3) 再运行 Rx_Part1_Generate_Data.m']);
end
sync_template_raw = DRoF_PCM_Parameters.SYNC_ref_after_ffe(:).';
disp('>>> 使用参数文件中的 SYNC_ref_after_ffe 作为同步模板。');

sync_template = sync_template_raw(:) - mean(sync_template_raw);
sync_len = length(sync_template);
best_phase = -1;
best_sync_start = -1;
best_score = -inf;
best_phase_rho = -inf;
best_phase_ratio = 0;
best_is_inverted = false;
best_corr_lags = [];
best_corr_out = [];
best_peak_abs = 0;

for phase_idx = 1:sps
    sym_phase = stream_full(phase_idx:sps:end);
    if length(sym_phase) < sync_len + pam_data_symbols
        continue;
    end

    % 只在前部窗口搜索同步峰，防止长数据区伪峰
    search_window = min(length(sym_phase), max(120000, 60 * sync_len));
    search_region = sym_phase(1:search_window);
    search_region_norm = search_region(:) - mean(search_region);

    [corr_out, corr_lags] = xcorr(search_region_norm, sync_template);
    [peak_abs, peak_idx] = max(abs(corr_out));
    sync_lag = corr_lags(peak_idx);
    sync_start_idx = sync_lag + 1;

    % 起点合法性检查
    if sync_start_idx < 1 || (sync_start_idx + sync_len - 1) > length(search_region)
        continue;
    end

    % 用归一化相关系数作为跨相位可比的打分
    seg = search_region(sync_start_idx:sync_start_idx + sync_len - 1);
    seg = seg(:) - mean(seg);
    denom = norm(seg) * norm(sync_template(:));
    if denom < 1e-12
        continue;
    end
    rho = (seg.' * sync_template(:)) / denom;  % 有符号相关系数
    score = abs(rho);

    % 计算该相位下的主次峰比，用于抑制伪峰
    abs_corr = abs(corr_out);
    [sorted_peaks, sorted_idx] = sort(abs_corr, 'descend');
    main_idx = sorted_idx(1);
    guard_bins = round(0.15 * sync_len);
    mask = true(size(abs_corr));
    left_i = max(1, main_idx - guard_bins);
    right_i = min(length(abs_corr), main_idx + guard_bins);
    mask(left_i:right_i) = false;
    if any(mask)
        second_peak = max(abs_corr(mask));
    else
        second_peak = 0;
    end
    peak_ratio = sorted_peaks(1) / max(second_peak, 1e-12);

    % 综合评分：相关强度 + 峰值歧义度
    composite_score = score * log(1 + peak_ratio);

    if composite_score > best_score
        best_score = composite_score;
        best_phase = phase_idx;
        best_sync_start = sync_start_idx;
        best_is_inverted = (rho < 0);
        best_corr_lags = corr_lags;
        best_corr_out = corr_out;
        best_peak_abs = peak_abs;
        best_phase_rho = score;
        best_phase_ratio = peak_ratio;
    end
end

if best_phase < 0
    error('联合同步失败：未找到合法同步峰。请检查VPI数据或同步模板。');
end

final_phase = best_phase;
final_sync_start = best_sync_start;
fprintf('>>> 联合同步完成: 最佳采样相位 = %d/%d, sync_start = %d, rho = %.4f, 主次峰比 = %.3f, 综合分数 = %.4f, 峰值 = %.2f, 极性反转 = %s\n', ...
    final_phase, sps, final_sync_start, best_phase_rho, best_phase_ratio, best_score, best_peak_abs, string(best_is_inverted));

% 在最终相位下采样，获取全部接收符号
all_symbols = stream_full(final_phase:sps:end);
fprintf('下采样后总符号数: %d (期望数据符号: %d)\n', length(all_symbols), pam_data_symbols);

% 绘制互相关图（最佳相位）
figure;
plot(best_corr_lags, abs(best_corr_out));
title_str = sprintf('SYNC xcorr at Best Phase (%s %s)', prbs_base_name, upper(run_mode));
title(title_str);
xlabel('Lag (symbols)'); ylabel('|Correlation|'); grid on;

% 极性恢复
if best_is_inverted
    disp('>>> 检测到接收信号极性反转，正在自动补偿...');
    all_symbols = -all_symbols;
else
    disp('>>> 信号极性正常。');
end

% ======================== 步骤4: 数据截取 ========================
if final_sync_start < 1
    error('帧同步异常: sync_start = %d < 1。', final_sync_start);
end
data_start_idx = final_sync_start + sync_len;  % 业务数据紧跟同步模板之后
data_end_idx = data_start_idx + pam_data_symbols - 1;

fprintf('数据起始索引: %d, 结束索引: %d (总符号: %d)\n', ...
    data_start_idx, data_end_idx, length(all_symbols));

% 安全检查
if data_end_idx > length(all_symbols)
    warning('数据末尾 (%d) 超出信号长度 (%d)，自动截断。', ...
        data_end_idx, length(all_symbols));
    rx_sym_aligned = all_symbols(data_start_idx : end);
else
    rx_sym_aligned = all_symbols(data_start_idx : data_end_idx);
end

% 确保提取的符号数不超过PAM数据长度
if length(rx_sym_aligned) > pam_data_symbols
    rx_sym_aligned = rx_sym_aligned(1:pam_data_symbols);
end

fprintf('数据已提取: %d 符号\n', length(rx_sym_aligned));

% ======================== 步骤5: 无标签同步质量验证 ========================
disp('步骤5: 验证同步质量(仅SYNC模板)...');

% 指标1：归一化相关分数（跨相位可比）
sync_score = best_score;

% 指标2：峰值歧义度（主峰/次峰）
abs_corr = abs(best_corr_out);
[sorted_peaks, sorted_idx] = sort(abs_corr, 'descend');
if isempty(sorted_peaks)
    error('同步失败：无法计算相关峰。');
end
main_idx = sorted_idx(1);
guard_bins = round(0.15 * sync_len);  % 主峰附近保护区，避免把同一峰旁瓣当次峰
mask = true(size(abs_corr));
left_i = max(1, main_idx - guard_bins);
right_i = min(length(abs_corr), main_idx + guard_bins);
mask(left_i:right_i) = false;
if any(mask)
    second_peak = max(abs_corr(mask));
else
    second_peak = 0;
end
ambiguity_ratio = sorted_peaks(1) / max(second_peak, 1e-12);

fprintf('>>> 同步质量(仅日志): sync_score = %.4f, 主次峰比 = %.3f\n', ...
    sync_score, ambiguity_ratio);

disp('>>> 数据流已成功提取并对齐。');

%% ===================== 准备DNN的输入和标签 ===========================
% 由于进行了 movmean  去均值，现在的信号中心为 0
% 直接将其峰值归一化到 [-1, 1] 区间以便神经网络输入
received_amp_norm = rx_sym_aligned / max(abs(rx_sym_aligned));

if strcmpi(run_mode, 'train')
    PAM_code_ideal_dnn = (PAM_code - 1.5) / 1.5;
end

%% === 5. 保存数据给NN (物理块严格对齐) ===
total_symbols = length(PAM_code); 
if length(received_amp_norm) > total_symbols
    received_amp_norm = received_amp_norm(1:total_symbols);
elseif length(received_amp_norm) < total_symbols
    total_symbols = length(received_amp_norm); 
    PAM_code = PAM_code(1:total_symbols); 
    if strcmpi(run_mode, 'train')
        PAM_code_ideal_dnn = PAM_code_ideal_dnn(1:total_symbols); 
    end
end

G = quant / Mm;
if mod(quant, Mm) ~= 0, error('quant 不能被 Mm 整除。'); end

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

pcm_ref = bp_OFDM(1:num_blocks);
pcm_ref_max = max(abs(pcm_ref));
if pcm_ref_max > 0
    pcm_ref_norm = pcm_ref / pcm_ref_max; 
else
    pcm_ref_norm = zeros(size(pcm_ref));
end

output_filename = sprintf('Data_For_NN_%s_%s_%d.mat', prbs_base_name, run_mode, received_optical_power);
full_output_filename = fullfile(mat_save_path, output_filename);

if strcmpi(run_mode, 'train')
    dnn_train_input = received_amp_norm; 
    dnn_train_label = PAM_code_ideal_dnn; 
    dnn_train_pcm_ref = pcm_ref_norm; 
    save(full_output_filename, 'dnn_train_input', 'dnn_train_label', 'dnn_train_pcm_ref');
    disp(['训练数据已保存: ', full_output_filename]);
elseif strcmpi(run_mode, 'test')
    dnn_test_input = received_amp_norm;
    original_pam_test_data = PAM_code; 
    dnn_test_pcm_ref = pcm_ref_norm; 
    save(full_output_filename, 'dnn_test_input', 'original_pam_test_data', 'dnn_test_pcm_ref');
    disp(['测试数据已保存: ', full_output_filename]);
end
disp('>> 对齐处理完成，现在可以安全进入 PyTorch！');

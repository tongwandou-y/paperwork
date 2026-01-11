% =========================================================================
% DRoF - PCM - FFE - PAM (v2 - Top-Down Automation)
% DRoF Tx
%==========================================================================
% 此代码参数基本设为自动的了，每次修改参数时只需要修改Part1部分以及Part2部分的Rs即可！！！！！
% 首先明确一点Fs_AWG = 64e9这个参数不能变，这是硬件决定的
% 其次fc = 10e9和quant = 8基本不变，也不会影响其他参数。
% 当修改系统目标符号速率时，根据目标符号速率和Fs_AWG 计算sps。比如目标符号速率为8Gsym/s，那么sps就为64 / 8 = 8。
% 此时，Rs_PAM_final、Required_PCM_Bitrate、fb会自动计算
% 然后，根据Rs的约束重新计算Rs的值。基本是由于fb变了导致的
% 此时，Rate、R也会自动计算
% --------------------------------------------------------------------------
% 每次修改 Part 后，都必须重新计算并设置一个新的 Rs！！！！！！
% 使其同时满足以下三个约束条件：
%     - 约束 1 (OFDM奈奎斯特): fb 必须能采样 Rs。
%       fb > 2 * Rs * (1 + Rolloff)
%     - 约束 2 (OFDM采样率): Rate 必须是整数。
%       Rate = Fs / Rs 必须是整数 (即 Rs 必须能被 Fs 整除)。
%     - 约束 3 (PCM采样率): R 必须是整数。
%       R = Fs / fb 必须是整数 (即 fb 必须能被 Fs 整除)。
% --------------------------------------------------------------------------
% 为了匹配 VPI TimeWindow 进行了手动补零
% 因此还需要设置total_sim_points这个参数！！！！！！
% 这个参数必须与 VPI TimeWindow 的值一样，且需要大于Rx_Part1_PreProcessing.m中的end_index变量
% --------------------------------------------------------------------------
% 文件名自动化：
% 1. 在 "核心设置区" 设置 PRBS 文件名 (例如 'PRBS15.txt')
% 2. 在 "核心设置区" 设置运行模式 ('train' 或 'test')
% 3. 脚本将自动生成带正确后缀 (如 _PRBS15_train) 的 .mat 和 .txt 文件
%==========================================================================  

clc;clear;close all;

%% === 基础设置区 ===
% --- 1. 设置统一的数据文件夹路径 ---
% 请在这里填入存放 PRBS31.txt, PRBS23.txt, PRBS31_pam4.txt 等所有文件的目录
data_dir = 'D:\paperwork\PRBS_Data';  % <--- 更换不同PRBS时，只需修改此处即可
                                
% --- 2. 选择本次要跑的 PRBS 文件名 ---
prbs_select_name = 'PRBS31.txt';           % 例如 'PRBS15.txt', 'PRBS23.txt', 'PRBS31.txt'

% --- 3. 自动生成 prbs_filename 完整路径 ---
prbs_filename = fullfile(data_dir, prbs_select_name); 
                                
% --- 4.从 'PRBS23.txt' 中提取 'PRBS23' ---
[~, prbs_base_name, ~] = fileparts(prbs_filename); % (例如 'PRBS23')

% --- 5. 设置运行模式 ---
%    'train': 用于生成训练集 (例如 PRBS23)
%    'test':  用于生成测试集 (例如 PRBS31)
run_mode = 'test';           % <--- 在这里手动设置为 'train' 或 'test'

% --- 6. 设置参数文件保存位置 ---
% 设置参数文件的目标保存目录
param_save_path = 'D:\paperwork\Experiment_Data\20Gsyms_20km\TX_Matlab_Param_mat';

% 检查目录是否存在，不存在则自动创建，防止报错
if ~exist(param_save_path, 'dir')
    mkdir(param_save_path);
    fprintf('检测到参数目录不存在，已自动创建: %s\n', param_save_path);
end

% --- 7. 设置 VPI 波形文件 (.txt) 保存位置 ---
vpi_save_path = 'D:\paperwork\Experiment_Data\20Gsyms_20km\VPI_Input_Data_txt'; 
if ~exist(vpi_save_path, 'dir'), mkdir(vpi_save_path); end

%% === Parameter Setting Part1(这部分参数是最高优先级的参数) ===
% 为了实现参数自动化，所有后续参数都将从这些值自动推导出来
% 所以，这里的参数需要是最终的、固定的实验目标
% 1. PAM4-RRC-AWG 模块
Fs_AWG = 60e9;            % 定义 AWG 采样率，GSa/s。在实际链路中，会首先看这个值，因为硬件的能力是固定的

% 2. RRC 模块
% 上采样率(每个符号的采样点数)
%       sps的设定依据是，当选择用一个 Fs_AWG Sa/s的AWG，去模拟一个Rs_PAM_final Gsym/s的信号时
%       必须用sps个采样点去描绘1个符号的波形
% sps需要是一个>=2的整数。Fs_AWG/sps 应该是一个 "G" 级的整数速率。
% 64/2=32G, 64/3=X, 64/4=16G, 64/5=X, 64/6=X, 64/7=X, 64/8=8G
% 这里选择 sps = 4 来获得一个 16 Gsym/s 的高速率。
sps = 3;                  % 目标：每个PAM4符号的采样点数

% 3. PCM 模块
quant = 8;                % PCM量化比特数 (8 bit/sample)  通常 3~15bit

% 4. PAM4 编码模块
MM = 4;                   % PAM调制阶数 (PAM4)

% 5. OFDM 模块
M = 16;                   % 设置QAM调制的阶数 (16-QAM)
Rolloff = 0.2;            % RRC滚降系数 (用于所有滤波器)
fc = 5e9;                 % OFDM载波频率，GHz
Fs = 4*fc;                % OFDM载波采样率，是载波频率的4倍，满足奈奎斯特采样定理，GSa/s0 

%% === Parameter Setting Part2 ===
% 这里的参数是基于Part1的参数，派生、计算出来的参数
% 1. RRC/AWG 派生参数
% 定义最终希望在光纤中传输的PAM4信号的符号速率（波特率），Gsym/s。这个值根据实验目标来设定
Rs_PAM_final = Fs_AWG / sps; % 64e9 / 4 = 16 Gsym/s
R_AWG = sps;              % AWG上采样倍率，必须与sps相同。这是为了确保信号的上采样倍率和为其设计的滤波器的参数相匹配

% 2. PAM4/PCM 派生参数
Mm = log2(MM);             % PAM4每个符号的比特数 (2 bit/sym)
Required_PCM_Bitrate = Rs_PAM_final * Mm; % PCM必须输出的比特率 (16e9 * 2 = 32 Gbps)
% 带通采样频率fb必须大于信号带宽(BandpassBw)的两倍，此时BandpassBw=2.4e9，因此fb至少为4.8e9，取5是为了让R=Fs/fb为整数
fb = Required_PCM_Bitrate / quant; % PCM的采样率 (32e9 / 8 = 4 GSa/s)

% 3. OFDM 派生参数 (必须适配 fb = 4 GSa/s)
% 奈奎斯特约束: 
%   为了让 fb 能对OFDM信号进行有效采样，
%   OFDM信号带宽 (BandpassBw) 必须 <= fb/2 (即 2 GHz)
%   BandpassBw = Rs * (1 + Rolloff) = Rs * 1.2
%   Rs * (1 + 0.2) <= 2e9  => Rs 必须 <= 1.667 Gsym/s
%【】【】【】【】【】【】【】【】【】【】【】【】【】
%
%
% 这里        需要满足最上方的三条限制
%
%【】【】【】【】【】【】【】【】【】【】【】【】【】
Rs = 1e9;               % 设定：QAM基带符号速率 (1.6 Gsym/s) (<= 1.667e9)

% 4. 采样率比率 (必须为整数)
Rate = Fs / Rs;             % OFDM上采样倍率 (40e9 / 1.6e9 = 25)
% bp sample number =bp sample frequency * data length /bit rate
R = Fs / fb;                % 带通采样倍数  PCM带通采样的降采样因子 (40e9 / 4e9 = 10)

% 5. 检查所有约束
if mod(Rate, 1) ~= 0 || mod(R, 1) ~= 0
    error('OFDM/PCM的采样率比率不是整数。R=%.2f, Rate=%.2f', R, Rate);
end
BandpassBw = Rs * (1 + Rolloff);
if fb < 2 * BandpassBw
    error('PCM采样率 fb (%.1fG) 低于OFDM信号的奈奎斯特频率 (%.1fG)。', fb/1e9, 2*BandpassBw/1e9);
end


%% === Parameter Setting Part3 ===
BitPerSym = log2(M);    % 计算每个QAM符号所承载的比特数
Rb = Rs * BitPerSym;    % 设置原始伪随机二进制序列（PRBS）的比特率 1e9表示1Gbps
dt=1/Fs;                % 计算采样点之间的时间间隔
N_Sc=510;               % 设置OFDM的子载波数量
SymPerCar=10;           % 设置每个子载波上承载的QAM符号数量 
N_IFFT=1024;            % 设置IFFT的点数为1024。这个值必须大于等于子载波数
CP = 1/8;               % 设置循环前缀的长度

%% ============== OFDM modulation ===============================
N_data=N_Sc*SymPerCar*log2(M);  % N_data表示本次仿真总共需要的比特数量

prbs_data = load(prbs_filename); % 加载数据

% 确保 prbs_data 是一个行向量，以便于 repmat 和后续的 vec2mat 处理
prbs_data_row = prbs_data(:)'; % 使用 (:) 确保其先变为一列，然后 ' 转置为一行

% 判断加载的比特流长度是否足够，如果不够，则重复该序列直到满足长度要求。
if length(prbs_data_row) < N_data
    % 已经是一个行向量，直接 repmat
    prbs_data_row = repmat(prbs_data_row, 1, ceil(N_data/length(prbs_data_row)));
end

Txdata = prbs_data_row(1:N_data);   % 截取所需长度的比特流作为本次发送的数据

% ====================== QAM modualtion =================================
Bin=vec2mat(Txdata,log2(M));	% 将一维的比特流 Txdata 转换成一个矩阵，每行 log2(M) 个比特，方便进行QAM映射
Dec=bi2de(Bin,'left-msb');		% 将每一行的二进制数转换为一个十进制数(对于64QAM转换为0~63的整数)
base_QAM=qammod(Dec',M,'gray');	% 对十进制序列进行64-QAM调制，并采用格雷码映射以降低误码率。
% ======================= S2P ==========================================
base_QAM1=reshape(base_QAM,N_Sc,SymPerCar).';  % serial to parallel
% ====================== ifft ==========================================
Subcarrier = (1:N_Sc) + (floor(N_IFFT/4) - floor(N_Sc/2));	% 计算数据子载波在1024点IFFT阵列中的具体位置索引
Conjugate_Sc = N_IFFT - Subcarrier + 2;	% 为了让IFFT变换后的时域信号是实数信号（而不是复数），计算共轭对称的子载波位置
IFFTmod = zeros(SymPerCar,N_IFFT); 
IFFTmod(:,Subcarrier) = base_QAM1;  % Subcarrier mapping here
IFFTmod(:,Conjugate_Sc)=conj(base_QAM1);% conjugate complex map 
IFFT=ifft(IFFTmod,N_IFFT,2);     % ifft
% ====================== Cyclic prefix ================================
CP_OFDM = [IFFT(:,N_IFFT*(1-CP)+1:N_IFFT) IFFT];	% 添加循环前缀。
% ====================== UpSample =====================================
Delay=10;           % 滤波器的群延迟或长度，单位是符号周期
delayOFDM = zeros(SymPerCar,floor(((1+CP)*N_IFFT+2*Delay)*Rate));	% 内存预分配
rcosOFDM = zeros(SymPerCar,floor((1+CP)*N_IFFT*Rate));

% 循环处理每一个OFDM符号
for i = 1:SymPerCar
	% 脉冲成形
    delayOFDM(i,:) = BasebandShaping(CP_OFDM(i,:),Fs,Rs,Rolloff,Delay);
    % 移除滤波器带来的群延迟
    rcosOFDM(i,:) = delayOFDM(i,Delay*Rate+1:end-Delay*Rate);
end
% ===================== Carrier Mod ===================================
symLen = 0:N_IFFT*(1+CP)*Rate-1;	% 创建一个时间序列，作为每个OFDM符号的离散时间索引
CarrierOFDM = zeros(SymPerCar,N_IFFT*(1+CP)*Rate);
for i = 1:SymPerCar
    CarrierOFDM(i,:) = rcosOFDM(i,:).*cos(2*pi*fc*symLen*dt);	% 上变频
end
% ====================== P2S ==========================================
stOFDM = zeros(1,SymPerCar*N_IFFT*(1+CP)*Rate);
for i = 1:SymPerCar
    stOFDM(1,(i-1)*(N_IFFT*(1+CP))*Rate+1:i*(N_IFFT*(1+CP))*Rate)= CarrierOFDM(i,:);
end
% ==================== Sample filter ===================================
BandpassBw = Rs * (1 + Rolloff);    % 带宽 = 符号速率 * (1 + 滚降系数)
Lf = fc-1/2*BandpassBw;
Uf = fc+1/2*BandpassBw;
OFDM=BandPF(stOFDM,Lf,Uf,Fs);
OFDM=OFDM/max(abs(OFDM));  % normalize
% =================== Bandpass sample ===============================
samplenumber=round(length(OFDM)/R);
bp_OFDM=zeros(1,samplenumber);
for m=1:samplenumber
    bp_OFDM(1,m)=OFDM(1,(m-1)*R+1);
end
%% =================== PCM  ==================
display(quant);
K=2^quant;                      % 量化电平的总数
V=fix(max(abs(bp_OFDM)))+1;     % 量化的电压范围
quantDelta = 2 * V/K;           % 每个量化电平之间的间隔（步长）
partition = (-V:quantDelta:V);
codebook = (-V - 1/2*quantDelta:quantDelta:V+1/2*quantDelta);
[~,quant_OFDM] = quantiz(bp_OFDM,partition,codebook);
% SQNR
SQNR=20*log10(norm(bp_OFDM)/norm(bp_OFDM-quant_OFDM));
display(SQNR);
% Mapping of quantized sequence code vectors to their corresponding indexes
quant_Dec = zeros(1,samplenumber); 
for i = 1:samplenumber
    for ii = 1:K
        if quant_OFDM(1,i) == codebook(1,ii)
            break;
        end
    end
    quant_Dec(1,i)=ii;
end
%% ==================  Coder ===================================
quant_Dec=quant_Dec-1;                      % 0~K-1 codebook Index
quant_Bin=de2bi(quant_Dec,'left-msb');      % dec to binary
NRZ_code=reshape(quant_Bin',samplenumber*quant,1);% NRZ Tx
% === 检查点 ===
% % 此时 NRZ_code 速率 = fb * quant = 4e9 * 8 = 32 Gbps.

%% ==================== PAM4 coding =============================

N_PAM=length(NRZ_code)/Mm;                % PAM coding length
Bin_code=reshape(NRZ_code,Mm,N_PAM)';
Dec_code=bi2de(Bin_code,'left-msb');      % to dec 
PAM_code=pammod(Dec_code,MM,0,'gray')';   % PAM4 modulation 注意：此处用的是格雷码！！！
PAM_code=(PAM_code+(MM-1))/2;             % 映射电平到 [0, 1, 2, 3]
% === 检查点 ===
% 此时 PAM_code 符号速率 = NRZ_code速率 / Mm = 32 Gbps / 2 = 16 Gsym/s.
% 这与我们的 Rs_PAM_final (16 Gsym/s) 目标完美匹配。

% --- 同步头 ---
% 动态构建同步头文件名 (例如 'PRBS31_pam4.txt')
sync_head_name = sprintf('%s_pam4.txt', prbs_base_name); 

% 组合完整路径：直接使用顶部的 data_dir + 文件名
sync_head_fullpath = fullfile(data_dir, sync_head_name);

fprintf('正在加载同步头文件: %s\n', sync_head_fullpath);

% 检查文件是否存在，方便排错
if ~exist(sync_head_fullpath, 'file')
    error('错误：找不到同步头文件！\n请确认文件 "%s" 是否在目录 "%s" 下。', sync_head_name, data_dir);
end

synchead = load(sync_head_fullpath)';        % 自动加载 PAM4 同步头，加载128个符号的PAM4同步序列
synchead1 = [synchead synchead];             % 组成256个符号的同步头，与发射端一致

PAM_code2 = [synchead1 0 0 PAM_code];       % 序列: [同步头, 两个0, PAM4数据]

rcos_PAM = [zeros(1,2000) PAM_code2];	% 在最前面添加一段零，作为保护间隔

%% ===================== FFE ====================================
% TapWeights 
Tap2 = [1 -0.45];
Tap3 = [0.075 0.75 0.175];
% Tap4 = [];
FFEMode = 1;
FFE = serdes.FFE('Mode',FFEMode, 'WaveType','Impulse', 'TapWeights',Tap3);
PAM_preE = FFE(rcos_PAM);

%% ===================== RRC ====================================
beta = 0.2; % RRC滤波器的滚降系数
N_span=12;  % 滤波器的符号截断数，N=12表示这个滤波器的长度覆盖12个符号的持续时间

RRCtx=rcosdesign(beta,N_span,sps);
up_PAM=upsample(PAM_preE ,R_AWG);
% === 检查点 ===
% up_PAM 采样率 = PAM_preE速率 * R_AWG = 16 Gsym/s * 4 = 64 GSa/s.
rcos_PAM=conv(up_PAM,RRCtx);
% rcos_PAM 采样率也是 64 GSa/s.

eyediagram(rcos_PAM(:,15000:20000),sps)

%% 手动补零以匹配 VPI TimeWindow
% 1. 定义与 VPI TimeWindow 一致的总点数
total_sim_points = 2200000; % 必须与 VPI TimeWindow (200000/64e9) 匹配

% 2. 手动在 rcos_PAM 后补零
current_len = length(rcos_PAM); % 这大概是 101,432
if current_len > total_sim_points
    error('信号长度大于VPI仿真窗口!');
end
pam_tx_padded = [rcos_PAM, zeros(1, total_sim_points - current_len)];

%% ===================== 保存参数和波形 (自动化命名) =============================

% --- 1. 根据 run_mode 确定文件后缀 ---
if strcmpi(run_mode, 'train')
    file_suffix = '_train';
elseif strcmpi(run_mode, 'test')
    file_suffix = '_test';
else
    error("无效的 'run_mode' 设置 (在脚本顶部)。请将其设置为 'train' 或 'test'。");
end

% --- 2. 保存参数 (.mat 文件) ---
DRoF_PCM_Parameters.OFDM_M = M;
DRoF_PCM_Parameters.OFDM_Rs = Rs;
DRoF_PCM_Parameters.OFDM_fc = fc;
DRoF_PCM_Parameters.OFDM_Fs = Fs;
DRoF_PCM_Parameters.OFDM_dt = dt;
DRoF_PCM_Parameters.OFDM_Rate = Rate;
DRoF_PCM_Parameters.OFDM_N_Sc = N_Sc;
DRoF_PCM_Parameters.OFDM_SymPerCar = SymPerCar;
DRoF_PCM_Parameters.OFDM_CP = CP;
DRoF_PCM_Parameters.OFDM_N_IFFT = N_IFFT;
DRoF_PCM_Parameters.OFDM_Txdata = Txdata;
DRoF_PCM_Parameters.OFDM_base_QAM = base_QAM;
DRoF_PCM_Parameters.OFDM_Subcarrier = Subcarrier;
DRoF_PCM_Parameters.OFDM_Rolloff = Rolloff;
DRoF_PCM_Parameters.OFDM_Delay = Delay;
DRoF_PCM_Parameters.OFDM_symLen = symLen;
DRoF_PCM_Parameters.OFDM_Lf = Lf;
DRoF_PCM_Parameters.OFDM_Uf = Uf;
DRoF_PCM_Parameters.OFDM = OFDM;
DRoF_PCM_Parameters.OFDM_fb = fb;
DRoF_PCM_Parameters.OFDM_R = R;
DRoF_PCM_Parameters.OFDM_samplenumber = samplenumber;
DRoF_PCM_Parameters.PCM_quant = quant;
DRoF_PCM_Parameters.PCM_codebook = codebook;
DRoF_PCM_Parameters.PAM_MM  = MM;
DRoF_PCM_Parameters.PAM_Mm  = Mm;
DRoF_PCM_Parameters.PAM_code  = PAM_code;
DRoF_PCM_Parameters.RRC_beta = beta;     % 保存滚降系数
DRoF_PCM_Parameters.RRC_N_span = N_span; % 保存滤波器跨度
DRoF_PCM_Parameters.RRC_R_AWG  = R_AWG;
DRoF_PCM_Parameters.RRC_sps  = sps;
DRoF_PCM_Parameters.rcos_PAM  = rcos_PAM;  % 这里不需要改，直接保存原始的、未补零的 rcos_PAM，因为 Rx 的 .mat 文件不需要补零！！
DRoF_PCM_Parameters.FFE_Taps = Tap3;
DRoF_PCM_Parameters.Fs_AWG = Fs_AWG;

% 构造 .mat 文件名 (例如 'DRoF_PCM_Parameters_PRBS15_train.mat')
param_filename = sprintf('DRoF_PCM_Parameters_%s%s.mat', prbs_base_name, file_suffix);

% 组合完整路径
param_full_path = fullfile(param_save_path, param_filename);

% 保存到指定路径
save(param_full_path, 'DRoF_PCM_Parameters'); 
fprintf('参数文件已保存到: %s\n', param_full_path);
% save('DRoF_PCM_Parameters.mat','DRoF_PCM');

% --- 3. 保存VPI波形 (.txt 文件) ---
% 将MATLAB生成的数字信号 幅度归一化和格式化，以符合VPI对外部导入信号文件的要求
% VPI 通常要求导入的信号是两列数据（时间列，幅度列），且幅度通常需要归一化到 [-1, 1] 或 [0, 1] 区间，以便驱动仿真中的调制器（MZM）或激光器。
pam_tx = 0.5 * pam_tx_padded / (sum(abs(pam_tx_padded))/ length(pam_tx_padded))+1;
pam_tx = pam_tx/2;  % 此时信号的中心为0.5
pam_tx = pam_tx./(max(abs(pam_tx)));
% ------------------------------------------------------------------
% 最终pam_tx为
%   - 最大值 (Max)：1.0
%   - 最小值 (Min)：大于 0 (通常在 0.1 ~ 0.2 左右，取决于信号的过冲程度)
%   - 中心点 (Bias)：大于 0.5。
% ------------------------------------------------------------------
dt_vpi = 1/Fs_AWG; % 采样时间间隔，用于告诉VPI每一个数据点代表多少时间
time_win=0:dt_vpi:(length(pam_tx)-1)*dt_vpi; % 生成时间轴向量
sequence=[time_win' pam_tx'];

% --- 4. 自动生成 VPI 文件名并指定保存路径 ---

% 使用 MM 变量创建 'PAM4' 字符串
pam_part = sprintf('PAM%d', MM);

% 组合成文件名 (例如 'PRBS15_PAM4_TX_20Gsym_train.txt')
% 注意：这里 rate_str_Gsym 是基于 Rs_PAM_final 计算的
rate_str_Gsym = Rs_PAM_final / 1e9; 
vpi_filename_only = sprintf('%s_%s_TX_%.0fGsym%s.txt', prbs_base_name, pam_part, rate_str_Gsym, file_suffix);

% 组合完整路径
vpi_full_path = fullfile(vpi_save_path, vpi_filename_only);

fprintf('正在保存VPI波形到: %s\n', vpi_full_path);

% 保存到指定路径
save(vpi_full_path, 'sequence', '-ascii');
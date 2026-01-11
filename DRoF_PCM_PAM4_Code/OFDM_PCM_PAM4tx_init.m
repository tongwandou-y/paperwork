% =========================================================================
% DRoF - PCM - FFE - PAM
% Designed by djb 2024/3/27
% Southwest Jiaotong University  
% DRoF Tx
%==========================================================================
clc;clear;close all;

%% Parameter Setting
M = 16;                 % 设置QAM调制的阶数
BitPerSym = log2(M);	% 计算每个QAM符号所承载的比特数
Rs = 2e9;               % 计算QAM基带信号的符号速率（波特率）
Rb = Rs * BitPerSym;	% 设置原始伪随机二进制序列（PRBS）的比特率 1e9表示1Gbps
fc = 10e9;              % 设置射频载波频率
Fs=4*fc;                % 设置载波的采样频率，是载波频率的4倍，满足奈奎斯特采样定理。
dt=1/Fs;                % 计算采样点之间的时间间隔
Rate=Fs/Rs;             % 计算插值（上采样）倍率，用于后续的脉冲成形
N_Sc=510;               % 设置OFDM的子载波数量
SymPerCar=2;            % 设置每个子载波上承载的QAM符号数量
N_IFFT=1024;            % 设置IFFT的点数为1024。这个值必须大于等于子载波数
CP = 1/8;               % 设置循环前缀的长度

%% ============== OFDM modulation ===============================
N_data=N_Sc*SymPerCar*log2(M);  % N_data表示本次仿真总共需要的比特数量

prbs_filename = 'PRBS23.txt'; % 更换不同PRBS时，只需修改此处即可

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
Bin=vec2mat(Txdata,log2(M));           % data shaping
Dec=bi2de(Bin,'left-msb');           % bin2dec
base_QAM=qammod(Dec',M,'gray');    % QAM mod（gray）
% ======================= S2P ==========================================
base_QAM1=reshape(base_QAM,N_Sc,SymPerCar).';  % serial to parallel
% ====================== ifft ==========================================
% Conjugate symmetric subcarrier mapping - Complex data IFFT point
Subcarrier = (1:N_Sc) + (floor(N_IFFT/4) - floor(N_Sc/2));
% Conjugate symmetric subcarrier mapping - Conjugate complex IFFT
Conjugate_Sc = N_IFFT - Subcarrier + 2;
IFFTmod = zeros(SymPerCar,N_IFFT); 
IFFTmod(:,Subcarrier) = base_QAM1;  % Subcarrier mapping here
IFFTmod(:,Conjugate_Sc)=conj(base_QAM1);% conjugate complex map 
IFFT=ifft(IFFTmod,N_IFFT,2);     % ifft
% ====================== Cyclic prefix ================================
CP_OFDM = [IFFT(:,N_IFFT*(1-CP)+1:N_IFFT) IFFT];
% ====================== UpSample =====================================
Rolloff = 0.2;      % RRC 
Delay=10;           % Root ascending cosine filter delay (or filter cutoff length)
delayOFDM = zeros(SymPerCar,floor(((1+CP)*N_IFFT+2*Delay)*Rate));
rcosOFDM = zeros(SymPerCar,floor((1+CP)*N_IFFT*Rate));

for i = 1:SymPerCar
    % baseband shaping
    delayOFDM(i,:) = BasebandShaping(CP_OFDM(i,:),Fs,Rs,Rolloff,Delay);
    % remove delay
    rcosOFDM(i,:) = delayOFDM(i,Delay*Rate+1:end-Delay*Rate);
end
% ===================== Carrier Mod ===================================
% Length range per carrier
symLen = 0:N_IFFT*(1+CP)*Rate-1;
CarrierOFDM = zeros(SymPerCar,N_IFFT*(1+CP)*Rate);
for i = 1:SymPerCar
    CarrierOFDM(i,:) = rcosOFDM(i,:).*cos(2*pi*fc*symLen*dt);
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

%RF pds
% OFDM1 = OFDM*1e2;
% showPsd(OFDM1,Fs); % 暂时注释掉绘图

% ===================  Bandpass sample ===============================
fb=5e9;     % 带通采样频率fb必须大于信号带宽(BandpassBw)的两倍，此时BandpassBw=2.4e9，因此fb至少为4.8e9，取5是为了让R=Fs/fb为整数
% bp sample number =bp sample frequency * data length /bit rate
R=Fs/fb;    % 带通采样倍数
samplenumber=round(length(OFDM)/R);
bp_OFDM=zeros(1,samplenumber);
for m=1:samplenumber
    bp_OFDM(1,m)=OFDM(1,(m-1)*R+1);   
end
%% =================== PCM  ==================
quant = 8;                      % quant bits (3~15bit)
display(quant);
K=2^quant;                      % quantization level
V=fix(max(abs(bp_OFDM)))+1;     % Voltage range of the signal
quantDelta = 2 * V/K;           % quantization space
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

%% ==================== PAM4 coding =============================
MM = 4;                                   % PAM Mod order
Mm=log2(MM);                              % mod bits
N_PAM=length(NRZ_code)/Mm;                % PAM coding length
Bin_code=reshape(NRZ_code,Mm,N_PAM)'; 
Dec_code=bi2de(Bin_code,'left-msb');      % to dec 
PAM_code=pammod(Dec_code,MM,0,'gray')';   % PAM4 modulation 注意：此处用的是格雷码！！！
PAM_code=(PAM_code+(MM-1))/2;             % 映射电平到 [0, 1, 2, 3]

% --- 同步头 ---
synchead = load('PRBS15_pam4.txt')';        % 加载 PAM4 同步头
synchead1 = [synchead synchead];

PAM_code2 = [synchead1 0 0 PAM_code];     % 序列: [同步头, 两个0, PAM4数据]
rcos_PAM = [zeros(1,2000) PAM_code2];     % add zeros 

%% ===================== FFE ====================================
% TapWeights 
Tap2 = [1 -0.45];
Tap3 = [0.075 0.75 0.175];
% Tap4 = [];
FFEMode = 1;
FFE = serdes.FFE('Mode',FFEMode, 'WaveType','Impulse', 'TapWeights',Tap3);
PAM_preE = FFE(rcos_PAM);

%% ===================== RRC ====================================
% 1. 定义 AWG 采样率。在实际链路中，会首先看这个值，因为硬件的能力是固定的
Fs_AWG = 32e9;
% 2. 定义最终希望在光纤中传输的PAM4信号的符号速率（波特率）。这个值根据实验目标来设定
Rs_PAM_final = 8e9; 
% 3. 上采样率(每个符号的采样点数)
%       sps的设定依据是，当选择用一个 Fs_AWG Sa/s的AWG，去模拟一个Rs_PAM_final 2 Gsym/s的信号时
%       必须用sps个采样点去描绘1个符号的波形
sps = Fs_AWG / Rs_PAM_final;    % sps的值就是通过上面两个参数设定的
% 4. 检查 sps 是否为整数 (这对于滤波器设计至关重要)
if mod(sps, 1) ~= 0
    error('sps (Fs_AWG / Rs_PAM_final) 必须为整数。请检查 Fs_AWG 和 Rs_PAM_final 的值。');
end
% 5. AWG上采样倍率，必须与sps相同。这是为了确保信号的上采样倍率和为其设计的滤波器的参数相匹配
R_AWG = sps;

beta = 0.5; % RRC滤波器的滚降系数
N_span=60;  % 滤波器的符号截断数，N=60表示这个滤波器的长度覆盖60个符号的持续时间

RRCtx=rcosdesign(beta,N_span,sps);
up_PAM=upsample(PAM_preE ,R_AWG);
rcos_PAM=conv(up_PAM,RRCtx);
% figure; plot(rcos_PAM); % 暂时注释掉绘图
rcos_PAM = awgn(rcos_PAM,80);
eyediagram(rcos_PAM(1:2000),sps)

%% ===================== 保存参数和波形 =============================
DRoF_PCM.OFDM_M = M;
DRoF_PCM.OFDM_Rs = Rs;
DRoF_PCM.OFDM_fc = fc;
DRoF_PCM.OFDM_Fs = Fs;
DRoF_PCM.OFDM_dt = dt;
DRoF_PCM.OFDM_Rate = Rate;
DRoF_PCM.OFDM_N_Sc = N_Sc;
DRoF_PCM.OFDM_SymPerCar = SymPerCar;
DRoF_PCM.OFDM_CP = CP;
DRoF_PCM.OFDM_N_IFFT = N_IFFT;
DRoF_PCM.OFDM_Txdata = Txdata;
DRoF_PCM.OFDM_base_QAM = base_QAM;
DRoF_PCM.OFDM_Subcarrier = Subcarrier;
DRoF_PCM.OFDM_Rolloff = Rolloff;
DRoF_PCM.OFDM_Delay = Delay;
DRoF_PCM.OFDM_symLen = symLen;
DRoF_PCM.OFDM_Lf = Lf;
DRoF_PCM.OFDM_Uf = Uf;
DRoF_PCM.OFDM = OFDM;
DRoF_PCM.OFDM_fb = fb;
DRoF_PCM.OFDM_R = R;
DRoF_PCM.OFDM_samplenumber = samplenumber;
DRoF_PCM.PCM_quant = quant;
DRoF_PCM.PCM_codebook = codebook;
DRoF_PCM.PAM_MM  = MM;
DRoF_PCM.PAM_Mm  = Mm;
DRoF_PCM.PAM_code  = PAM_code; % ！！！！ 修改：变量名 OOK_code -> code
DRoF_PCM.RRC_beta = beta;     % 保存滚降系数
DRoF_PCM.RRC_N_span = N_span; % 保存滤波器跨度
DRoF_PCM.RRC_R_AWG  = R_AWG;
DRoF_PCM.RRC_sps  = sps;
DRoF_PCM.rcos_PAM  = rcos_PAM;
DRoF_PCM.Fs_AWG = Fs_AWG;

save('DRoF_PCM_Parameters.mat','DRoF_PCM');

% --- 保存VPI波形 ---
% 归一化
pam_tx = 0.5 * rcos_PAM / (sum(abs(rcos_PAM))/ length(rcos_PAM))+1; 
pam_tx = pam_tx/2;
pam_tx = pam_tx./(max(abs(pam_tx)));
dt_vpi = 1/Fs_AWG; 
time_win=0:dt_vpi:(length(pam_tx)-1)*dt_vpi;
sequence=[time_win' pam_tx'];

% --- 自动生成 VPI 文件名 ---
% 1. 从 'PRBS23.txt' 中提取 'PRBS23'
[~, prbs_name, ~] = fileparts(prbs_filename);
% 2. 使用 MM 变量创建 'PAM4' 字符串
pam_part = sprintf('PAM%d', MM);
% 3. 组合成最终文件名
vpi_filename = sprintf('%s_%s_TX.txt', prbs_name, pam_part);

fprintf('正在保存VPI波形到: %s\n', vpi_filename);
save(vpi_filename, 'sequence', '-ascii');
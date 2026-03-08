% =========================================================================
% DRoF - PCM - FFE -PAM
% Designed by djb 2024/3/27
% Southwest Jiaotong University  
% Rx
%==========================================================================
%% ======================== Parameter Load ======================
% load ('ExpData2\PCM_OOK\DRoF_PCM_OOK_4bit.mat')
% M = DRoF_PCM.OFDM_M;
% Rs = DRoF_PCM.OFDM_Rs;
% fc = DRoF_PCM.OFDM_fc;
% Fs = DRoF_PCM.OFDM_Fs;
% dt = DRoF_PCM.OFDM_dt;
% Rate = DRoF_PCM.OFDM_Rate;
% N_Sc = DRoF_PCM.OFDM_N_Sc;
% SymPerCar = DRoF_PCM.OFDM_SymPerCar;
% CP = DRoF_PCM.OFDM_CP;
% N_IFFT = DRoF_PCM.OFDM_N_IFFT;
% Txdata = DRoF_PCM.OFDM_Txdata;
% base_QAM = DRoF_PCM.OFDM_base_QAM;
% Subcarrier = DRoF_PCM.OFDM_Subcarrier;
% Rolloff = DRoF_PCM.OFDM_Rolloff;
% Delay = DRoF_PCM.OFDM_Delay;
% symLen = DRoF_PCM.OFDM_symLen;
% Lf = DRoF_PCM.OFDM_Lf;
% Uf = DRoF_PCM.OFDM_Uf;
% OFDM = DRoF_PCM.OFDM;
% fb = DRoF_PCM.OFDM_fb;
% R = DRoF_PCM.OFDM_R;
% samplenumber = DRoF_PCM.OFDM_samplenumber;
% quant = DRoF_PCM.PCM_quant;
% codebook = DRoF_PCM.PCM_codebook;
% MM  = DRoF_PCM.PAM_MM;
% Mm  = DRoF_PCM.PAM_Mm;
% OOK_code  =DRoF_PCM.PAM_OOK_code;
% R_AWG  = DRoF_PCM.RRC_R_AWG;
% sps  = DRoF_PCM.RRC_sps;
% rcos_OOK  = DRoF_PCM.rcos_OOK;
% Fs_AWG = DRoF_PCM.Fs_AWG;
% beta_rrc = DRoF_PCM.RRC_beta;
% span_rrc = DRoF_PCM.RRC_N_span;

%% ===================== data received ==========================
% received=readwaveform();
received=rcos_OOK;
% test = load('ExpData2\PCM_OOK\4bit\B2B\-16dBm_1.mat');
% test1 = test.received;
% received=[test1 0];
received = [received 0];
% save('ExpData2\PCM_OOK\6bit\10km_1\-4dBm_RF.mat', 'received');
% -------------- Simulation  Exp ----------------------
% save('Sim\B2B_16QAM_PAM4\-9dB.mat', 'received');
% load('Experiment\16qam\-10adB.mat');
%---------------------------------

% Normailize
% -----------------------------------------------------------------
received = received-mean(received);
received = 0.5*received/(sum(abs(received))/length(received))+1; 
received = received/2; 
% -----------------------------------------------------------------
% eyediagram
eyediagram(received(1,10000:15000),6);
%% ========================= Resample  ==========================
fs = Fs_AWG;     % AWG Fs
% waveform_RX = resample(received, fs, Fs_OSC);
%% =================  De-synchronization code  ==================
Baudrate = fs / sps;  % PAM-Baud rate 
synchead = load('PRBS_ook.txt')';  % 128
synchead = [synchead synchead];
synch = repmat(synchead,8,1);
synch1 = reshape(synch,1,size(synch,1)*size(synch,2));
% index=zeros(1,floor(length(received)))/2;
% for i1=1:floor(length(received))/2  %128 is sync header length 
%      index(i1)=sum((received(i1:i1+2*128*fs/Baudrate-1)).*synch1);
% end
% if(1)
%     figure;plot(abs(index)); title('Synchronization peak for x pol.');
% end
% [~, head]=max(abs(index)); % start point 
% -------------------------------------------------
[frameLoc,xcorrPlot] = frameSyn(received,synch1);
plot(xcorrPlot);
title('Frame Syn');
% -------------------------------------------------
head1 = frameLoc+(2*128+2).*fs/Baudrate;
EqTrainLen = 1000;  % eq length
stream = received(1,head1:head1+(samplenumber*quant/Mm+EqTrainLen)*fs/Baudrate-1); 

%% ===================== PAM recovery ===========================
streamFilter = LowPF(stream,-1*Baudrate,Baudrate,fs);
streamAmp = abs(streamFilter);
% downSample --------------
for i = 1:fs/Baudrate
    receivedAmp = streamAmp(1,i:fs/Baudrate:end);
    OOKref = [OOK_code(1:1000) OOK_code];
    [SNR(i),AmpDecision(:,i)] = OOKCalSNR(receivedAmp,OOKref);
end
[MaxSNR,SamIndex] = max(SNR);
receivedAmp1 = streamAmp(1,SamIndex:fs/Baudrate:end);
%% ==================== post dfe ============================
% Create a PAM4 modulator System object
hMod = comm.PAMModulator(MM);  
% Create a DFE equalizer that has 10 feed forward taps and five feedback
% taps. The equalizer uses the LMS update method with a step size of 0.01. 
numFFTaps =30;   numFBTaps = 10;
equalizerDFE = dfe(numFFTaps,numFBTaps,rls(0.99,1));  
% modulator reference constellation. The reference constellation is
% determined by using the |constellation| method. For decision directed
% operation, the DFE must use the same signal constellation as the
% transmission scheme.
equalizerDFE.SigConst = constellation(hMod).';   
% Equalize the signal to remove the effects of channel distortion.
received_amp2=receivedAmp1*2-1; 
OOK_code1=OOK_code(1:1000)*2-1;        %-1 1形式
[received_eq,~] = equalize(equalizerDFE,received_amp2,OOK_code1);  
received_eq=received_eq(EqTrainLen+1:end); % 去均衡导频
received_eq=(received_eq+1)/2;           % 0 1 形式
% eyediagram
eyediagram(received_eq(1,1:1000),3);

% decision
% SamIndex = 2;
OOKre=OOKdecision(received_eq',AmpDecision,SamIndex);
%% ======================  PAM4 decoder =========================
% SER calculation
OOK_re1 = (OOKre+1)/2;
[SERnum,SERratio] = symerr(OOK_code,OOK_re1);
display(SERratio);
% display(SERnum);
% pam4 demod
de_PAM1=pamdemod(OOKre,MM,0,'gray');     % PAM4 demodulation
Bin1=de2bi(de_PAM1,'left-msb');            % dec to bin
Rx_code1=reshape(Bin1',quant*samplenumber,1);
%% ======================  decode ===============================
quant_Bin1=reshape(Rx_code1',quant,samplenumber)'; % S2P
quant_Dec1=bi2de(quant_Bin1,'left-msb');    % bin 2 dec
quant_Dec1=quant_Dec1+1;                    % 0~K-1 to 1~K
quant_OFDM_re=zeros(samplenumber,1);
for ii=1:samplenumber
    jj=quant_Dec1(ii,1);
    quant_OFDM_re(ii,1)=codebook(1,jj);     % 恢复的量化码
end
% QSNR Cal
for m=1:samplenumber
    bp_OFDM_Tx(1,m)=OFDM(1,(m-1)*R+1);   
end
quant_OFDM_re_temp = quant_OFDM_re';


Power_QuantSignal = mean(abs(quant_OFDM_re).^2);
Power_TxSignal = mean(abs(bp_OFDM_Tx).^2);
QuantNoise = bp_OFDM_Tx-quant_OFDM_re_temp;
Power_QuantNoise = mean(abs(QuantNoise).^2);
PCM_SQNR = 10*log10(Power_QuantSignal/Power_QuantNoise);
% SQNR = 10*log10(Power_TxSignal/Power_QuantNoise);
% display(SQNR);
display(PCM_SQNR);


% Up sample
DA_code=zeros(length(OFDM),1);
for i=1:samplenumber
    DA_code(R*(i-1)+1,1)=quant_OFDM_re(i,1);     
end

%% ======================restuct RF signal========================
% Sampling discrete signals with zero-recovery insertion
re_OFDM = BandPF(DA_code',Lf,Uf,Fs);

%% ======================OFDM demodulation========================
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
    % Match Filter
    delay_OFDM(ii,:)=MatchFilter(reCarrier_OFDM(ii,:),Fs,Rs,Rolloff,Delay);
    % delay
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

% scatterplot(re_QAM1);
% plot
xyz(:,1)=real(re_QAM1);
xyz(:,2)=imag(re_QAM1);
z3=round(rand(1,length(re_QAM1))*log2(M))+1;%与x的长度相等的向量
xyz(:,3)=reshape(z3,log2(M)*length(re_QAM1)/6,1);
datamin=min(min(xyz(:,1)),min(xyz(:,2)));
datamax=max(max(xyz(:,1)),max(xyz(:,2)));
datamin = floor(datamin);
datamax = ceil(datamax);
xyz(:,3)=xyz(:,3)- xyz(:,3);
sizeXYz = size(xyz);
searchR =0.2;  %搜索圆的半径
for i=1:sizeXYz(1)
    index_i = find(xyz(:,1)> xyz(i, 1)- searchR & xyz(:, 1)< xyz(i, 1)+ searchR ...
        &xyz(:,2)> xyz(i,2)- searchR & xyz(:,2)< xyz(i,2)+ searchR);%对散点进行分类
    sizeIndexI= size(index_i);
    xyz(i,3)= sizeIndexI(1);
end
[sortXYz,sortI]= sort(xyz(:, 3));
sz=10;
scatter(xyz(sortI, 1),xyz(sortI,2),sz,xyz(sortI, 3),'filled');
hold on;
grid on;
box on;
%------------------------------------------------------------------------------------------------------------------------------
% map = addcolorplus(317);
% colormap(map);
% scatplot(real(re_QAM1),imag(re_QAM1),'circles',sqrt((range(real(re_QAM1))/100)^2 + (range(imag(re_QAM1))/100)^2),300,2,1,15);
% set(gca,'XLim',[-4 4],'YLim',[-4 4],'xtick',-4:1:4,'ytick',-4:1:4,...
%     'XTickLabel',{},'YTickLabel',{})
% set(gca,'XColor','none','YColor','none')
% set(gca,'PlotBoxAspectRatio',[1 1 1]);
% title('Constellation');
% colorbar off
% box on;
% grid on;
% QAM EVM
% --------------------------------------------------------------------------------------------------------------------------------
evm=comm.EVM;
rmsEVM=step(evm,base_QAM.',re_QAM1.');
display(rmsEVM);

% OFDM BER
re_base=qamdemod(re_QAM1,M,'gray');
re_base1=de2bi(re_base','left-msb');
re_base2=reshape(re_base1',1,length(Txdata));
[~,BERratio]=biterr(Txdata,re_base2');
display(BERratio);

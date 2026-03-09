function [SNR,AmpDecision]=OOKCalSNR(receivedAmp,OOK_code)
% ---------------------------------------------------
% OOK SNR calculation
% Input:
%      receivedAmp: signal amplitude
%      OOK_code: signal tx signal
% Output: 
%      SNR : OOK SNR
%      CenAmp : 0~1 decision threshold
%  designed by Deng Jiongbin in 2024/4/15
%  Southwest Jiaotong Univertsy
% ---------------------------------------------------
zero_index = find(OOK_code==0);
one_index = find(OOK_code==1);

mean_zero_amp = mean(receivedAmp(1,zero_index));
mean_one_amp = mean(receivedAmp(1,one_index));

CenAmp = (mean_zero_amp + mean_one_amp)/2;

signal_power = ((mean_zero_amp-CenAmp).^2)*length(zero_index)+((mean_one_amp-CenAmp).^2)*length(one_index);

noise_zero_power = sum(abs(receivedAmp(zero_index)-mean_zero_amp).^2);
noise_one_power = sum(abs(receivedAmp(one_index)-mean_one_amp).^2);

SNR = 10*log10(signal_power/(noise_zero_power+noise_one_power));
AmpDecision = CenAmp;
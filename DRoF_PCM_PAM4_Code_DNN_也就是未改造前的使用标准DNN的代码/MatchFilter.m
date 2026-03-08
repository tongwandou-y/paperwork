function [y]=MatchFilter(data,fs,Rs,rolloff,Delay) 
%---------------------------------
% Designed by Deng Jiongbin  Southwest Jiaotong University
% 2024/3/28
%---------------------------------
% Square RRC for matching filter and dowmSample
% data：baseband signal
% fs：Fs carrier 
% Rs：baseband signal symbole rate
% rolloff： rolloff coeffs
% Delay：filter delay coeff

%% Match filter 
% H=rcosine(Rs,fs,'sqrt',rolloff,Delay); 
H = rcosdesign(rolloff,Delay*2,fs/Rs,'sqrt');
match_I=conv(real(data),H); 
match_Q=conv(imag(data),H);
%% down Sample
Rate=fs/Rs;  
down_I=downsample(match_I,Rate); 
down_Q=downsample(match_Q,Rate);
y=down_I+1j*down_Q;
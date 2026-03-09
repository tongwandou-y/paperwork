function [dataOut] = BasebandShping(dataIn,Fs,Rs,rolloff,Delay)
% ========================================================
% designed by Jiongbin Deng Southwest Jiaotong University
%  e-mail: 815234256@qq.com
% ===============================================
% RRC: upsample shaping filter
% data：baseband data
% fs：carrier sample frequency
% Rs：baseband symbol rate
% rolloff：rolloff coeff
% Delay：filter delay factor
I = real(dataIn);
Q = imag(dataIn);
UpRate=Fs/Rs; % 过采样率
upI = upsample(I,UpRate);
upQ = upsample(Q,UpRate);
% baseband shaping filter
H=rcosine(Rs,Fs,'sqrt',rolloff,Delay);
% H = rcosdesign(rolloff,Delay*2,fs/Rs,'sqrt');
rcosI = conv(upI,H);
rcosQ = conv(upQ,H);
dataOut = rcosI+1i*rcosQ;
end


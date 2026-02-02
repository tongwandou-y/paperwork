function [dataOut] = BandPF(dataIn,Lf,Uf,Fs)
% ======================================
% Bandpass filter
% dataIn：signal sample point
% Lf: filter lower frequency
% Uf: filter Upper frequency
% Fs: Sample frequency
% ======================================
[m,n] = size(dataIn);
if(m~=1)
    error('input stream error!');
end
Fstream = fftshift(fft(dataIn));
filter = zeros(1,n);
% filter(1,((Lf+Fs/2)/Fs*n+1):(Uf+Fs/2)/Fs*n) = 1;
filter(1, round((Lf+Fs/2)/Fs*n+1) : round((Uf+Fs/2)/Fs*n)) = 1;
Fstream = Fstream.*filter;
dataOut = ifft(ifftshift(Fstream));
dataOut=real(dataOut);

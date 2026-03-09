function stream_out = LowPF(stream_in,lower_f,upper_f,fsa)
% Version 2.1 Designed by Ye Jia 2013/8/22
% Low pass filter for RF signal

[m,n] = size(stream_in);
if(m~=1)
    error('input stream error!');
end
Fstream = fftshift(fft(stream_in));
filter = zeros(1,n);
% filter(1,((lower_f+fsa/2)/fsa*n+1):(upper_f+fsa/2)/fsa*n) = 1;
% 修复索引非整数警告，并限制在合法范围内
idx_start = round((lower_f+fsa/2)/fsa*n + 1);
idx_end   = round((upper_f+fsa/2)/fsa*n);
idx_start = max(1, min(n, idx_start));
idx_end   = max(1, min(n, idx_end));
filter(1, idx_start : idx_end) = 1;

Fstream = Fstream.*filter;
stream_out = ifft(ifftshift(Fstream));

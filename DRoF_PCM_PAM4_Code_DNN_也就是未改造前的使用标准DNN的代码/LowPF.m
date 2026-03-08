function stream_out = LowPF(stream_in,lower_f,upper_f,fsa)
% Version 2.1 Designed by Ye Jia 2013/8/22
% Low pass filter for RF signal

[m,n] = size(stream_in);
if(m~=1)
    error('input stream error!');
end
Fstream = fftshift(fft(stream_in));
filter = zeros(1,n);
filter(1,((lower_f+fsa/2)/fsa*n+1):(upper_f+fsa/2)/fsa*n) = 1;
Fstream = Fstream.*filter;
stream_out = ifft(ifftshift(Fstream));

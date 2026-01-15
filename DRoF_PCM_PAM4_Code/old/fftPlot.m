function [f,ydB] = fftPlot(data,Fs,xlim)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
N_plot = length(data);
x = data;
y = 2 * fft(x);
y = y(1:floor(N_plot/2)) / N_plot;
ydB = 10 .* log10(abs(y).^2 ./ 50) + 30;
f = [0:N_plot/2-1]./N_plot .* Fs;

figure;
plot(f/1e9, ydB, 'linewidth', 2,'color','b');
set(gca,'xlim',xlim);
grid on;
xlabel('频率 [GHz]');
ylabel('功率 [dBm]');

end

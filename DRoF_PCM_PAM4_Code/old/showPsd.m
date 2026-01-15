function showPsd(sig, sample_rate)
[PW,f] = periodogram(sig,hanning(length(sig)),length(sig),sample_rate,'power','centered');
f = f/1e9;

PdBm = 10*log10(PW*1000);
figure; plot(f, PdBm);
hold on;
xlabel('Frequency [GHz]');
ylabel('Power [dBm]');
set(gca,'XLim',[0 10])
set(gca,'XTick',[0:1:10]);
set(gca,'XTickLabel',[0:1:10])
set(gca,'YLim',[-80 0])
set(gca,'YTick',[-80:20:0])
set(gca,'YTickLabel',[-80:20:0])
grid on;
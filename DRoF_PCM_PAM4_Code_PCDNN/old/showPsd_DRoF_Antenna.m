function showPsd_DRoF_Antenna(sig, sample_rate)
[PW,f] = periodogram(sig,hanning(length(sig)),length(sig),sample_rate,'power','centered');
f = f/1e9;

PdBm = 10*log10(PW*1000);
figure; plot(f, PdBm);
hold on;
xlabel('Frequency [GHz]');
ylabel('Power [dBm]');
set(gca,'XLim',[4.5 5.5])
set(gca,'XTick',[4.5:0.2:5.5]);
set(gca,'XTickLabel',[4.5:0.2:5.5])
set(gca,'YLim',[-100 -20])
set(gca,'YTick',[-100:20:-20])
set(gca,'YTickLabel',[-100:20:-20])
grid on;
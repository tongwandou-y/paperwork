function [frame_location, xcorr_plot] = frameSyn(rx_sig, preamble)
    % frameSyn: 鲁棒的帧同步函数 (使用全局最大值)
    
    % 1. 计算互相关
    temp = xcorr(rx_sig, preamble);
    
    % 2. 归一化 (仅用于绘图，不影响找最大值)
    xcorr_plot = abs(temp);
    if max(xcorr_plot) > 0
        xcorr_plot = xcorr_plot ./ max(xcorr_plot);
    end
    
    % 3. 寻找全局最大值 (最可靠的方法)
    [~, max_idx] = max(xcorr_plot);
    
    % 4. 转换为起始索引
    % xcorr 的滞后与索引换算: lag = index - length(rx_sig)
    % 我们需要的是信号中 preamble 开始的位置
    frame_location = max_idx - length(rx_sig) + 1;
end


% function [frame_location, xcorr_plot] = frameSyn(rx_sig, preamble)
% temp = xcorr(rx_sig, preamble);
% xcorr_plot = abs(temp)./max(abs(temp));
% % plot(xcorr_plot);
% height_threshold = 0.99;
% [pks,locs] = findpeaks(xcorr_plot,'MinPeakHeight',height_threshold);
% frame_location = locs(pks==max(pks))-length(rx_sig)+1;

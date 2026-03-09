function [frame_location, xcorr_plot, is_inverted] = frameSyn(rx_sig, preamble)
    % frameSyn: 鲁棒的帧同步函数 (修复极性反转与假峰问题)
    
    % 1. 强制转为行向量
    rx_sig = rx_sig(:).';
    preamble = preamble(:).';
    
    % 2. 极其关键：分别去直流！(这能彻底消除平坦区假峰)
    rx_sig = rx_sig - mean(rx_sig);
    preamble = preamble - mean(preamble);
    
    % 3. 计算互相关
    temp = xcorr(rx_sig, preamble);
    xcorr_plot = temp; % 保存真实互相关图用于观察
    
    % 4. 寻找绝对值最大的峰 (因为光信号可能被反相，产生巨大负峰)
    [~, max_idx] = max(abs(temp)); 
    
    % 5. 检查极性：如果最大峰处的值是负数，说明信号极性反转了！
    if temp(max_idx) < 0
        is_inverted = true;
    else
        is_inverted = false;
    end
    
    % 6. 计算偏移量
    frame_location = max_idx - length(rx_sig);
end

% function [frame_location, xcorr_plot] = frameSyn(rx_sig, preamble)
%     % frameSyn: 鲁棒的帧同步函数 (使用全局最大值)
%     
%     % 1. 计算互相关
%     temp = xcorr(rx_sig, preamble);
%     
%     % 2. 归一化 (仅用于绘图，不影响找最大值)
%     xcorr_plot = abs(temp);
%     if max(xcorr_plot) > 0
%         xcorr_plot = xcorr_plot ./ max(xcorr_plot);
%     end
%     
%     % 3. 寻找全局最大值 (最可靠的方法)
%     [~, max_idx] = max(xcorr_plot);
%     
%     % 4. 转换为起始索引
%     % xcorr 的滞后与索引换算: lag = index - length(rx_sig)
%     % 我们需要的是信号中 preamble 开始的位置
%     frame_location = max_idx - length(rx_sig) + 1;
% end

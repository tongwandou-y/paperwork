function [y]=OOKdecision(receivedAmp,decision_Amp,SamIndex)
% =========================================================================
% PAM8 decision
% Designed by djb 2024/3/28
% Southwest Jiaotong University  
% Rx
%==========================================================================
% received_Amp:Magnitude of sample data 
% decision_Amp:Center level of 0~1 levels of the OOK signal

decision_Amp = decision_Amp(1,SamIndex);


L_OOK=length(receivedAmp);
y=zeros(1,L_OOK);
for i = 1:L_OOK
    if (receivedAmp(i,1)<=decision_Amp)
         y(1,i)=-1;
    else 
         y(1,i)=1;
    end
end
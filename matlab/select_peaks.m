function [selected_peaks] = select_peaks(sig,peaks,fs)
% thr = 0.01;

n = length(peaks);
selected_peaks = ones(1,length(peaks));
selected_peaks(1) = 0;
selected_peaks(end) = 0;

amp_diff_mean = mean(diff(sig(peaks)));
amp_diff_std = std(diff(sig(peaks)));
thr = median(sig(peaks));

for i = 2:n-1
    check1  = check_normal_beat2( peaks(i-1),peaks(i),peaks(i+1), median(diff(peaks)),fs,0.5);
    if check1 == 0 || sig(peaks(i)) < thr*0.5 || sig(peaks(i)) > thr*3  % || abs( sig(peaks(i)) - sig(peaks(i-1)) - amp_diff_mean ) > 2*amp_diff_std || abs( sig(peaks(i)) - sig(peaks(i+1)) - amp_diff_mean ) > 2*amp_diff_std         
        selected_peaks(i) = 0;
    end
end
% selected_peaks = selected_peaks(selected_peaks~=0);

end

function [ out ] = check_normal_beat2( beat1,beat2,beat3, rr,fs,r_ratio)
% 세 beat의 정보를 이용하여 정상적인 beat인지를 확인

% r_ratio=0.3;
r=2*abs( ((beat1-2*beat2+beat3)/fs) / ( (beat1-beat2)/fs*(beat1-beat3)/fs*(beat2-beat3)/fs ) );
    if r<0.5 && abs((beat2-beat1) - rr) <  rr*r_ratio && abs((beat3-beat2) - rr) <  rr*r_ratio
        out=1;
    else
        out=0;
    end
    
end

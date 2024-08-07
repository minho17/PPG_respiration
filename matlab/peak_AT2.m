function [peak_max peak_min filtered_sig] = peak_AT2(sig,fs,flag_plot)
% PPG peak detector
% ref. Adaptive threshold method for the peak detection of photoplethysmographic waveform
% addpath('C:\Users\minho\Documents\minho\reserch\bio_signal\ECG\project_DDD\matlab codes\sleep_detector\func_for_sd')
n=length(sig);
peak_max=zeros(1,floor(n/2));
peak_min=zeros(1,floor(n/2));
max_thr=zeros(1,n);
min_thr=zeros(1,n);

%%% Signal conditioning using discrete cosine transform (DCT) % normal BPF now...
filtered_sig=filter_bf(sig,0.5,9,fs);

%%% Adaptive threshold detection (ATD)
std_ppg=std(filtered_sig(1:fs*10));
max_Sr=-0.6;
min_Sr=0.6;
% max_slope=0.2*max(filtered_sig);
% min_slope=0.2*min(filtered_sig);
% de_x=ceil(fs*0.01);
max_thr(1:2)=filtered_sig(1:2);
min_thr(1:2)=filtered_sig(1:2);
correct_inter=0.6*fs;
pre_max_peak=0;
pre_min_peak=-2*correct_inter;
max_possible=0;
min_possible=0;
max_count=1;
min_count=1;

for i=3:n
    % slope update
    flag_max_detected=0;
    if max_count==1
        max_thr(i)=0.2*max(filtered_sig(1:fs*10));
    else
        max_thr(i)=max_thr(i-1)+max_Sr*(abs(filtered_sig(peak_max(max_count-1)))+std_ppg)/fs;
    end
    if min_count==1
        min_thr(i)=0.2*min(filtered_sig(1:fs*10));
    else
        min_thr(i)=min_thr(i-1)+min_Sr*(abs(filtered_sig(peak_min(min_count-1)))+std_ppg)/fs;
    end

    % signal 상황 별 변화
    if (filtered_sig(i) < filtered_sig(i-1)) && (filtered_sig(i-1) > filtered_sig(i-2)) % 위 꼭지점
        if max_possible==1
            peak_max(max_count)=i-1;
            max_count=max_count+1;
            flag_max_detected=1;
            pre_max_peak=i;
        end
        max_possible=0;        
    elseif (filtered_sig(i) > filtered_sig(i-1)) && (filtered_sig(i-1) < filtered_sig(i-2)) % 아래 꼭지점
        if min_possible==1
            peak_min(min_count)=i-1;
            min_count=min_count+1;
            pre_min_peak=i;
        end               
        min_possible=0;       
    elseif (filtered_sig(i) >= filtered_sig(i-1)) && (filtered_sig(i-1) >= filtered_sig(i-2)) % 계속 상승
        if (max_thr(i-1)<=filtered_sig(i))&&(i-pre_max_peak > correct_inter)
            max_thr(i)=filtered_sig(i);
            max_possible=1;
        end       
    elseif (filtered_sig(i) <= filtered_sig(i-1)) && (filtered_sig(i-1) <= filtered_sig(i-2)) % 계속 하락
        if (min_thr(i-1)>=filtered_sig(i))&&(i-pre_min_peak > correct_inter)
            min_thr(i)=filtered_sig(i);
            min_possible=1;
        end
    end 
    
    % peak correction interval update
    if flag_max_detected==1
        if max_count>2
            correct_inter=0.6*mean(diff(peak_max(1:max_count-1)));
        end
    end
    
end
peak_max=peak_max(peak_max~=0);
peak_min=peak_min(peak_min~=0);

if flag_plot > 0
    f = figure(flag_plot);
    f.Position = [5 5 1300 1000];
    t = (1:length(sig))/(fs*60);
    ax3(1)=subplot(3,1,1);
    plot(t,sig); title('Raw PPG');
    ax3(2)=subplot(3,1,2);
    plot(t,filtered_sig); title('Filtered PPG');
    ax3(3)=subplot(3,1,3);
    plot(t,filtered_sig)
    hold on;
    plot(peak_max/(fs*60),filtered_sig(peak_max),'r*')
    plot(peak_min/(fs*60),filtered_sig(peak_min),'g*')
    plot(t,max_thr,'y-')
    plot(t,min_thr,'k-')
    xlabel('Time (min)');
    title('Peak detection');
    hold off;
    linkaxes(ax3,'x');
end

end


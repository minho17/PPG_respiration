
clc;
close all;
clear all;

flag_data = 1; % 1=Capno / 2=BIDMC
fs = 30;

if flag_data == 1
    path_data = 'C:\Users\USER\Desktop\minho\PPG_resp\data\CapnoBase\data\mat\';
    list = dir([path_data '*.mat']);
    n_data = length(list); 
    
    for i =1:n_data
        load([path_data list(i).name]);

        sig_resp = signal.co2.y;
        sig_ppg = signal.pleth.y;

        if i == 1
            fs0 = param.samplingrate.pleth; 
        end

        Wn = [0.1, 15] * 2/fs0;
        N = 3; 
        [a,b] = butter(N,Wn);
        sig_resp = filtfilt(a,b,sig_resp);
        sig_ppg = filtfilt(a,b,sig_ppg);

        sig_resp = resample(sig_resp,fs,fs0);
        sig_ppg = resample(sig_ppg,fs,fs0);

        if i == 1
            n_sig = size(sig_ppg,1);
            data = zeros(n_data,n_sig,2);
        end

        data(i,:,2) = sig_resp; 
        data(i,:,1) = sig_ppg; 
    end

    file_name = 'data1_Capno.mat';
elseif flag_data ==2 
    path_data = 'C:\Users\USER\Desktop\minho\PPG_resp\data\BIDMC_1.0.0\bidmc_csv\';
    list = dir([path_data '*_Signals.csv']);
    n_data = length(list); 
    fs0 = 125;

    for i =1:n_data
        sig_all = readtable([path_data list(i).name],'VariableNamingRule','preserve');
        sig_resp = sig_all.RESP;
        sig_ppg = sig_all.PLETH;

        Wn = [0.1, 15] * 2/fs0;
        N = 3; 
        [a,b] = butter(N,Wn);
        sig_resp = filtfilt(a,b,sig_resp);
        sig_ppg = filtfilt(a,b,sig_ppg);

        sig_resp = resample(sig_resp,fs,fs0);
        sig_ppg = resample(sig_ppg,fs,fs0);

        if i == 1
            n_sig = size(sig_ppg,1);
            data = zeros(n_data,n_sig,2);
        end

        data(i,:,2) = sig_resp; 
        data(i,:,1) = sig_ppg; 
    end

    file_name = 'data1_BIDMC.mat';
end

save(file_name,'fs','data');

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
    
def peak_AT2(sig,fs,flag_plot,path_save=0):
    n = len(sig)
    peak_max = np.zeros( (int(n/2)),dtype = np.int32 )
    peak_min = np.zeros( (int(n/2)),dtype = np.int32 )
    max_thr = np.zeros( (n) ) 
    min_thr = np.zeros( (n) )

    N = 3
    btype = "bandpass"
    W1 = 0.5/ (fs / 2)
    W2 = 5 / (fs / 2)
    sos = signal.butter(N, [W1, W2], btype, output="sos")
    filtered_sig = signal.sosfiltfilt(sos, sig)
    # filtered_sig = sig

    std_ppg = np.std(filtered_sig[:fs*10])
    max_Sr = -0.7
    min_Sr = 0.37
    max_thr[0:2] = filtered_sig[0:2]
    min_thr[0:2] = filtered_sig[0:2]
    # correct_inter = 0.6*fs
    correct_inter = 0
    pre_max_peak = 0
    pre_min_peak = -2*correct_inter
    max_possible = 0
    min_possible = 0
    max_count = 0
    min_count = 0

    for i in range(2,n):
        flag_max_detected = 0
        if max_count == 0:
            max_thr[i] = 0.2*np.max(filtered_sig[0:fs*10])
        else:
            max_thr[i] = max_thr[i-1] + max_Sr*(np.abs(filtered_sig[int(peak_max[max_count-1])])+std_ppg)/fs
        
        if min_count == 0:
            min_thr[i] = 0.2*np.min(filtered_sig[0:fs*10])
        else:
            min_thr[i] = min_thr[i-1] + min_Sr*(np.abs(filtered_sig[int(peak_min[min_count-1])])+std_ppg)/fs

        if (filtered_sig[i] < filtered_sig[i-1]) and (filtered_sig[i-1] > filtered_sig[i-2]):
            if max_possible == 1:
                peak_max[max_count] = i-1
                max_count = max_count+1
                flag_max_detected = 1
                pre_max_peak = i

            max_possible=0;        
        elif (filtered_sig[i] > filtered_sig[i-1]) and (filtered_sig[i-1] < filtered_sig[i-2]):
            if min_possible==1:
                peak_min[min_count] = i-1
                min_count = min_count+1
                pre_min_peak = i
                     
            min_possible=0;       
        elif (filtered_sig[i] >= filtered_sig[i-1]) and (filtered_sig[i-1] >= filtered_sig[i-2]):
            if (max_thr[i-1] <= filtered_sig[i]) and (i-pre_max_peak > correct_inter):
                max_thr[i] = filtered_sig[i]
                max_possible = 1
        
        elif (filtered_sig[i] <= filtered_sig[i-1]) and (filtered_sig[i-1] <= filtered_sig[i-2]):
            if (min_thr[i-1] >= filtered_sig[i]) and (i-pre_min_peak > correct_inter):
                min_thr[i] = filtered_sig[i]
                min_possible = 1
     
        if flag_max_detected == 1:
            if max_count > 1:
                correct_inter = 0.6*np.mean(np.diff(peak_max[0:max_count]))
                a=1

    peak_max = peak_max[0:max_count]
    peak_min = peak_min[0:min_count]

    if flag_plot > 0:
        plt.figure(flag_plot)
        t = np.arange(0, len(sig), 1)/(fs*60)
        plt.subplot(3,1,1)
        plt.plot(t,sig)
        plt.title('Raw PPG')
        plt.subplot(3,1,2)
        plt.plot(t,filtered_sig)
        plt.title('Filtered PPG')
        plt.subplot(3,1,3)
        plt.plot(t,filtered_sig)
        plt.plot(peak_max/(fs*60),filtered_sig[peak_max],'r*')
        plt.plot(peak_min/(fs*60),filtered_sig[peak_min],'g*')
        plt.plot(t,max_thr,'y-')
        plt.plot(t,min_thr,'k-')
        plt.xlabel('Time (min)')
        plt.title('Peak detection')
        # plt.savefig(path_save + '/peaks.png')
        plt.show()

    return peak_max, peak_min, filtered_sig

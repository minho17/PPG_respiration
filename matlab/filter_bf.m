function [out]= filter_bf(sig,fre1,fre2,fs)

if nargin < 4, fs=360; end
%  n=3;
%  wn=fre1;
%  fn=fs/2;
%  
%  ftype='high';
%  [b, a]=butter(n,wn/fn,ftype);
%  out=filter(b,a,sig);
%  
%  wn=fre2;
%  ftype='low';
%  [b, a]=butter(n,wn/fn,ftype);
%  out=filter(b,a,out);
 
 Wn=[fre1 fre2]*2/fs; % cutt off based on fs
 N = 3; % order of 3 less processing
 [a,b] = butter(N,Wn); %bandpass filtering
 out = filtfilt(a,b,sig);
 
end


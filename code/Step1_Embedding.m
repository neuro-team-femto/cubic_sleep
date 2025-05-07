clc,clear all;close all;

%selected subject with clear wake-to-sleep transition
subs = [1,2,5,6,9,15,18,20,22,24,26,27,30,32,33,34,35,40,41,42]; 

addpath(genpath('./utility/letswave7-master'));


sub_folder = ['..\..\data\LX_EEG\data1\'];
ls = dir([sub_folder,'ds*lw6']);
fs = 100;
t_start = 1*fs;
t_end = 1800*fs;
t = linspace(t_start,t_end,(t_end - t_start + 1)) / fs;

% Wavelet parameter
wavelet_name = 'cmor1-1.5';
central_freq = centfrq(wavelet_name);
freq_start = 0.5;
freq_end = 20;
num_bin = 200;
w = linspace(freq_start,freq_end,num_bin)';
scales = (central_freq/(1/fs))./w;
delta_band = find( 0.5 < w & w < 4);
alpha_band = find( 8 < w & w < 12);

figure('Position',1.0e+03*[  0.1106    0.1778    1.2712    0.4842]);
end_win = zeros(length(subs),1);

for s_idx = 1:length(ls)
    sub = subs(s_idx);
    data_file = [sub_folder,ls(subs(s_idx)).name];
    lwdata = FLW_load.get_lwdata('filename',data_file);
    data = squeeze(lwdata.data(:,1,:,:,:,t_start:t_end));
    
    % Wavelet transformation
    amp_wav = abs(cwt(data',scales,wavelet_name));
    amp_ratio = sum(amp_wav(delta_band,:),1) ./ sum(amp_wav(alpha_band,:),1);
    sm_ratio= smoothdata(amp_ratio,"gaussian",300);

    % 3 min larger than 1
    thres_start = 1;
    dur_tran_start = 60*fs;
    cond = sm_ratio > thres_start;
    % Find the start and end indices of consecutive true sequences
    d = diff([0, cond, 0]);

    % This finds the start and the end of transition period
    run_starts = find(d ==  1);        
    run_ends = find(d == -1);     
    % Calculate the durations of the runs
    run_dur = run_ends - run_starts + 1;          % Length in samples
    req_run_idx = find(run_dur >= dur_tran_start);
    t_transtart = run_starts(req_run_idx(1));


    thres_end = 1;
    dur_tran_end = 120*fs;
    cond = sm_ratio > thres_end;
    % Find the start and end indices of consecutive true sequences
    d = diff([0, cond, 0]);

    % This finds the start and the end of transition period
    run_starts = find(d ==  1);        
    run_ends = find(d == -1);     
    % Calculate the durations of the runs
    run_dur = run_ends - run_starts + 1;          % Length in samples
    req_run_idx = find(run_dur >= dur_tran_end);
    t_tranend = run_starts(req_run_idx(1));
    t_ana = [t_transtart-200*fs, t_tranend+200*fs];

    % It's possible 200s before the onset of transitions is already out of the beginning of the recording
    if t_ana(1) < 0
        wav_ana = amp_wav(:,[1:t_ana(2)]);
        t_ana_axis = [1:t_ana(2)]/fs;
    else
        wav_ana = amp_wav(:,[t_ana(1):t_ana(2)]);
        t_ana_axis = [t_ana(1):t_ana(2)]/fs;
    end

    % Extract the first SVD mode in wake and sleep state separately
    wav_wake = wav_ana(:,1:60*fs);
    [Uw,Sw,Vw] = svd(wav_wake,'econ');
    wav_sleep = wav_ana(:,(end-60*fs):end);
    [Us,Ss,Vs] = svd(wav_sleep,'econ');
   

    % Make sure the mode is always positive
    if Uw(:,1) < 0
        mode_wake = -1*Uw(:,1);
    else
        mode_wake = Uw(:,1);
    end

    if Us(:,1) < 0
        mode_sleep = -1*Us(:,1);
    else
        mode_sleep = Us(:,1);
    end

    % Project normalized spectrogram to the difference of wake_mode and sleep_mode
    wav_norm = wav_ana ./ vecnorm(wav_ana, 2, 1);
    mu = (wav_norm-mode_sleep)'*(mode_wake-mode_sleep) / norm(mode_wake - mode_sleep,2).^2;
    
    % Scale mu to [-1,1]
    res_mu_norm = 2*mu_norm - 1;

    % Mild smoothing serves as low-pass filtering before down-sampling.
    res_mu_sm = smoothdata(res_mu_norm,"gaussian",100);

    % Down-sampling avoid over-fitting and reduce computational load.
    % efficiency 
    step = 10;
    ds_mu = res_mu_sm(1:step:end);
    if ~exist('./mu', 'dir')
        mkdir('./mu');
    end
    filename = ['./mu/',ls(subs(s_idx)).name(23:27),'.mat';];
    save(filename,'ds_mu');

end

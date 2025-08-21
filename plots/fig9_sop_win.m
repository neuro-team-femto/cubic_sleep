clc,clear all;close all;

sub_folder = ['../data/EEG/'];
ls = dir([sub_folder,'ds*mat']);

% Focus on the first half hours
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

figure('Position',1.0e+03*[0.1818    0.1610    1.2144    0.5576]);
for s_idx = 1:length(ls)
    subplot(4,5,s_idx);

    data_file = [sub_folder,ls(s_idx).name];
    load(data_file)
    data = squeeze(data(:,1,:,:,:,t_start:t_end));
    
    % Take the Wavelt representation
    amp_wav = abs(cwt(data',scales,wavelet_name));

    amp_ratio = sum(amp_wav(delta_band,:),1) ./ sum(amp_wav(alpha_band,:),1);
    sm_ratio= smoothdata(amp_ratio,"gaussian",300);

    % 1 min larger than 1
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

    % %% Extract dominant mode in wake and sleep state
    if t_ana(1) < 0
        wav_ana = amp_wav(:,[1:t_ana(2)]);
        t_ana_axis = [1:t_ana(2)]/fs;
    else
        wav_ana = amp_wav(:,[t_ana(1):t_ana(2)]);

        %  
        t_ana_axis = [t_ana(1):t_ana(2)]/fs;
    end

    sm_wav = smoothdata(abs(amp_wav),2,"gaussian",300);
    hdl = imagesc(gca,t,w, sm_wav);hold on;
    hdl.Parent.YDir = 'normal';
    colormap turbo;
    set(gca,'clim',[-8,30]);
    set(gca,'ylim',[1,15]);
    ylabel('Frequency [Hz]');
    xlabel('Time [s]');  
    plot([t_transtart,t_transtart]/fs,[1,15],'k','LineWidth',2);
    plot([t_tranend,t_tranend]/fs,[1,15],'k','LineWidth',2);
    set(gca,'xlim',[t(1),t_ana_axis(end)+200]);
    area(gca,[t_ana_axis(1),t_ana_axis(1)+60], [15, 15], 'FaceColor', 'blue', 'FaceAlpha', 0.3);
    area(gca,[t_ana_axis(end)-60,t_ana_axis(end)], [15, 15], 'FaceColor', 'blue', 'FaceAlpha', 0.3);
    if t_ana_axis(1)-50 > 0
        set(gca,'xlim',[t_ana_axis(1)-50,t_ana_axis(end)+50]);
    else
        set(gca,'xlim',[t_ana_axis(1),t_ana_axis(end)+50]);
    end
    tit = ['sub',num2str(s_idx)];
    title(gca,tit,'FontSize',12);
end
saveas(gcf,'../figures/fig9_sOP_win.png');

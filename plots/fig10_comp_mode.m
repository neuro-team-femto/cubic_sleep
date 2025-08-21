%% Compare the second dominant mode of SVD on the whole windows and the mode_wake - mode_sleep
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
    % end_win(s_idx) = t_tranend/fs;
    % save('end_win.mat','end_win');
    % %% Extract dominant mode in wake and sleep state
    if t_ana(1) < 0
        wav_ana = amp_wav(:,[1:t_ana(2)]);
        t_ana_win = [1:t_ana(2)]/fs;
    else
        wav_ana = amp_wav(:,[t_ana(1):t_ana(2)]);
        t_ana_win = [t_ana(1):t_ana(2)]/fs;
    end
    % Dominant mode in wake state
    wav_wake = wav_ana(:,1:60*fs);
    [Uw,Sw,Vw] = svd(wav_wake,'econ');
    wav_sleep = wav_ana(:,(end-60*fs):end);
    [Us,Ss,Vs] = svd(wav_sleep,'econ');
    % wav_tran = wav_ana(:,150*fs:(end-200*fs));
    % [U_tran,S_tran,V_tran] = svd(wav_tran,'econ');
    

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
    [U,S,V] = svd(wav_ana,'econ');
    if U(:,1) < 0
        U1 = -U(:,1);
        V1 = -V(:,1);
    else
        U1 = U(:,1);
        V1 = U(:,1);
    end

    if mu(1)*V(1,2)<0
        U2 = -U(:,2);
        V2 = -V(:,2);
    else 
        U2 = U(:,2);
        V2 = V(:,2);
    end
    if s_idx == 15
        U2 = -1*U2;
    end
    if s_idx == 18
        U2 = -1*U2;
    end
    subplot(4,5,s_idx);
    freq = w;
    plot(freq,mode_wake-mode_sleep,'LineWidth',1.5);hold on;plot(freq,U2,'LineWidth',1.5);
    xlabel('Frequency [Hz]','FontSize',10);
    tit = ['sub',num2str(s_idx)];
    title(gca,tit,'FontSize',12);
    ylabel('Mode','FontSize',10);

    if s_idx == 19
        leg = legend({'$U_w^1-U_s^1$', '$U_2$'}, ...
            'Interpreter','latex','FontSize',12);
        legend boxoff;
        set(leg,'Position',[0.82 0.1 0.05 0.2]) % [x y w h] normalized
    end
end

saveas(gcf,'../figures/fig10_comp_mode.png');
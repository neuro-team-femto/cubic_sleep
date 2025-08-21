%% Embedding extraction (sub 9)
clc,clear all;close all;
embed_folder = ['..\data\EEG\'];
ls = dir([embed_folder,'ds*mat']);

% Focus on first 30 minutes
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

% Define frequency band of interests
delta_band = find( 0.5 < w & w < 4);
alpha_band = find( 8 < w & w < 12);
 
% Plot settings
figure('Position',1.0e+03*[  0.1106    0.1778    1.2712    0.4842]);
fig = figure('Position',[441.0000  125.0000  586.4000  503.2000]);
for k = 1:3
    axe{k} = axes('Position',[0.15,0.1+0.27*(k-1),0.7,0.18]);
end
axe{4} = axes('Position',[0.03,0.1+0.27*(2-1),0.05,0.18]);
axe{5} = axes('Position',[0.92,0.1+0.27*(2-1),0.05,0.18]);

% Load sample subject
s_idx = 9;
data_file = [embed_folder,ls(s_idx).name];
load(data_file);
data = squeeze(data(:,1,:,:,:,t_start:t_end));

% Take the Wavelt representation
amp_wav = abs(cwt(data',scales,wavelet_name));

% Decide the transition window
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

% Extract dominant mode in wake and sleep state
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

% Extract the low_dimensional dynamics
wav_norm = wav_ana ./ vecnorm(wav_ana, 2, 1);
mu = (wav_norm-mode_sleep)'*(mode_wake-mode_sleep) / norm(mode_wake - mode_sleep,2).^2;
mu = smoothdata(mu,"gaussian",100);

gca = axe{3};
sm_wav = smoothdata(abs(wav_ana),2,"gaussian",300);
hdl = imagesc(gca,t_ana_win,w, sm_wav);hold(gca,"on");
hdl.Parent.YDir = 'normal';
colormap turbo;
set(gca,'clim',[-8,30]);
set(gca,'ylim',[1,15]);

ylabel('Frequency [Hz]');    
set(gca,'xlim',[t_ana_win(1),t_ana_win(end)]);
hold(gca,'on');
ylim = get(gca,'ylim');
t_end = t(run_starts(req_run_idx(1)))+ 2*60;
wake_time = [t_ana_win(1),t_ana_win(1)+60];
sleep_time = [t_ana_win(end)-60,t_ana_win(end)];
area(gca,wake_time, [15, 15], 'FaceColor', 'blue', 'FaceAlpha', 0.3);
area(gca,sleep_time, [15, 15], 'FaceColor', 'blue', 'FaceAlpha', 0.3);
set(gca, 'XTick', []);
title(gca,'EEG SOP Spectrogram','FontSize',10);
ylabel(gca,'Frequency [Hz]','FontSize',10);
text(gca,0.06, 0.3, ["Wake"], 'Units', 'normalized',...
     'FontSize', 11, 'Color', 'k', 'HorizontalAlignment', 'center');
text(gca,0.94, 0.8, ["Sleep"], 'Units', 'normalized',...
     'FontSize', 11, 'Color', 'k', 'HorizontalAlignment', 'center');

gca = axe{1}
mu_scale = (mu - min(mu)) ./ (max(mu) - min(mu));
recon_spec = mode_wake*mu_scale'+ mode_sleep*(1-mu_scale)';
sm_recon = smoothdata(recon_spec,2,'gaussian',300);
hdl = imagesc(gca,t_ana_win,w,recon_spec); %
hdl.Parent.YDir = 'normal';
colormap turbo;
set(gca,'ylim',[1,15]);
set(gca,'fontsize',10);
title(gca,'Reconstruction (\mu(t)U_w^{1}+(1-\mu(t))U_s^{1})','FontSize',10);
xlabel(gca,'Time [s]');
set(gca,'clim',[0,0.15]);
ylabel(gca,'Frequency [Hz]','FontSize',10);

gca = axe{2};
plot(gca,t_ana_win,mu);
set(gca,'xlim',[t_ana_win(1),t_ana_win(end)]);
set(gca, 'XTick', []);
title(gca,'\mu(t) (projection on U_w^{1}-U_s^{1})','FontSize',10);
set(gca,'fontsize',10);

% Plot two patches for the wake and sleep mode 
dummy_width = 5;
mode_wake_patch = repmat(mode_wake, 1, dummy_width);
mode_sleep_patch = repmat(mode_sleep, 1, dummy_width);

% 3) Create a new axis for the wake mode patch
gca =  axe{4};
imagesc(gca, 1:dummy_width, w, mode_wake_patch);
set(gca, 'YDir','normal', 'XTick',[], 'YTick',[]);
title(gca, 'U_w^{1}', 'FontSize', 9);
colormap(gca, turbo);        % Use the same colormap as your main spectrogram
clim(gca, [0, 0.16]);       % Match color limits if you want
set(gca,'ylim',[1,15]);
ylabel(gca,'Frequency [Hz]','FontSize',10);


gca =  axe{5};
imagesc(gca, 1:dummy_width, w, mode_sleep_patch);
set(gca, 'YDir','normal', 'XTick',[], 'YTick',[]);
title(gca, 'U_s^{1}', 'FontSize', 9);
colormap(gca, turbo);        % Use the same colormap as your main spectrogram
clim(gca, [0, 0.16]);       % Match color limits if you want
set(gca,'ylim',[1,15]);
set(gca,'xlim',[1,5]);
ylabel(gca,'Frequency [Hz]','FontSize',10);

saveas(gcf,'..\figures\fig2a_sample_embedding.png');

%% Plot all embeddings
clc,clear all;close all;
folder = ['../embedding/'];
figure('Position', 1.0e+03 * [0.1146    0.1074    1.2304    0.5408]);
ls = dir([folder,'*.mat']);

fs = 10;
for f_idx = 1:length(ls)
    filename = [folder,ls(f_idx).name];
    load(filename);
    subplot(4,5,f_idx);
    t = [1:length(ds_mu)]/fs;
    plot(t,ds_mu);
    tit = ['sub',num2str(f_idx)];
    title(tit);
    set(gca,'xlim',[t(1),t(end)]);
    set(gca,'fontsize',10)
end
saveas(gcf,'..\figures\fig2b_all_embeddings.png');
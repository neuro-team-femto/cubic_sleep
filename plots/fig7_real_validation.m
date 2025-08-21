clc,clear all;close all;

%load the embedding
load('..\embedding\sub19.mat');
% Download MCCM_sub08.mat from https://zenodo.org/records/16907957 and put
% into the folder  ./data/Inference/
load('../data/Inference\MCMC_sub8.mat');
load('mode_wake.mat');
load('mode_sleep.mat');

min_KL_idx = find(KL==min(KL));
min_KL_traj = pp_sample(:,min_KL_idx);
min_RMSE_idx = min_RMSE_idx+1;
mode_KL_idx = mode_KL+1;
map_idx = Joint_MAP_idx + 1;
x_map = pp_sample(:,map_idx);
mode_KL_traj = pp_sample(:,mode_KL_idx);
x_pick_RMSE = pp_sample(:,min_RMSE_idx);

load('wav_sub08.mat');
set(0, 'DefaultTextFontName', 'Arial');  
set(0,'DefaultAxesFontName', 'Arial');  
set(0, 'DefaultTextFontName', 'Arial');  
set(0,'DefaultAxesFontName', 'Arial');  
t = [0:length(ds_mu)-1]'/10;

%Re-scale mu and model output to [0,1]
mu_scale = (ds_mu - min(ds_mu)) ./ (max(ds_mu) - min(ds_mu));


blue = [0.2980	0.4471	0.6902];
green = [0.3333	0.6588	0.4078];
red = [0.7686	0.3059	0.3216];
gray  = [0.4 0.4 0.4];
fig = figure('Position',[297.8000   89.8000  802.4000  652.0000]);
axs = {};
widths = [0.34,0.4];
for i = 1:2
    width = widths(i);
    for j = 1:4
        axs{i,j}= axes('Position',[0.12+(i-1)*0.47,0.1+(j-1)*0.23,width,0.15])
    end
end



%% Plot a small random subset of trajectory and show the x(t) with minimal KL value
gca = axs{1,1};
c_real = lines;
Xq10  = prctile(pp_sample,10,2);          
Xq90  = prctile(pp_sample,90,2);
f = fill(gca,[t; flipud(t)], [Xq10; flipud(Xq90)], [0.8 0.8 1], ...
    'FaceAlpha',0.5,'EdgeColor','none');hold(gca,'on')
real = plot(gca,t,ds_mu,'Color',red,'LineWidth',1.5);
best = plot(gca,t,x_map,'Color',gray,'LineWidth',1.5);hold(gca,'on');
leg = legend(gca,best, '$x_{MAP}(t)$', ...
     'Location','best', 'Interpreter', 'latex');
leg.Box ='off';
leg.FontSize = 10;
title(gca,'Model output (Joint MAP)');
xlabel(gca,'Time [s]'); 
set(gca,'FontSize',10);
set(gca,'xlim',[t(1),t(end)]);
set(gca,'ylim',[0-2.5,2.5]);
%% Plot a small random subset of trajectory and show the x(t) with minimal RMSE value
gca = axs{1,2};
f = fill(gca,[t; flipud(t)], [Xq10; flipud(Xq90)], [0.8 0.8 1], ...
    'FaceAlpha',0.5,'EdgeColor','none');hold(gca,'on')
real = plot(gca,t,ds_mu,'Color',red,'LineWidth',1.5);hold(gca,'on');
best = plot(gca,t,x_pick_RMSE,'Color',blue,'LineWidth',1.5);
leg = legend(gca,best, '$x_{min\_RMSE}(t)$', ...
     'Location','best', 'Interpreter', 'latex');
leg.Box ='off';
leg.FontSize = 10;
title(gca,'Model output (Minimal RMSE)');
set(gca,'FontSize',10);
set(gca,'xlim',[t(1),t(end)]);
set(gca,'ylim',[-2.5,2.5]);
%% Plot ksdensity for the minimal KL case
gca = axs{1,3};
% compute ksdensity
x_pick_KL = pp_sample(:,min_KL_idx);
grid = [-2.5:0.001:2.5];
[model_pdf, grid] = ksdensity(x_pick_KL,grid);
[data_pdf,grid] = ksdensity(ds_mu',grid);
kl_divergence = min(KL);
fill(gca,grid,data_pdf, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');hold(gca,'on'); % Class 1, red shaded area
h= fill(gca,grid,model_pdf ,'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');% Class 1, red shaded area
set(h, 'FaceColor', blue);
tit = ['Minimal KL = ',num2str(kl_divergence,'%.2f')];
title(gca,tit);
leg = legend(gca, {'$\mu(t)$', '${x}(t)$'}, 'Interpreter', 'latex');
leg.FontSize = 12;
leg.Box = 'off';
set(gca,'FontSize',10);
%% KL distribution
KL_map= KL(map_idx);
RMSE_map = RMSE(map_idx);
ax = axs{1,4};               
% cla(ax)                      % start from a clean slate
KL_grid = [0:0.01:2];
RMSE_grid = [0.2:0.01:1.5];% --- build a common grid ---------------------------------------------------

% --- kernel densities ------------------------------------------------------
[kl_pdf,   grid_KL]   = ksdensity(KL, KL_grid,'Support','positive','Bandwidth',0.04);hold(gca,"on");
[rmse_pdf, grid_RMSE] = ksdensity(RMSE,RMSE_grid,'Support','positive','Bandwidth',0.04);
KL_mode = grid_KL(find(kl_pdf == max(kl_pdf)));
RMSE_mode = grid_RMSE(find(rmse_pdf == max(rmse_pdf)));

% --- plot on LHS y-axis: KL-divergence -------------------------------------
yyaxis(ax,'left')
kl_fill = fill(ax,grid_KL,kl_pdf, ...
               'b','FaceAlpha',0.50, ...
               'EdgeColor',  'none', ...
               'DisplayName','KL PDF');hold(ax,'on')
set(kl_fill, 'FaceColor', green);
ylims = ylim(ax);
kl_mode_line = plot(ax,[KL_mode  KL_mode], [ylims(1) ylims(2)],'--', 'Color','k','LineWidth', 1.5);
ylabel(ax,'Density (KL)');
hold(ax,'on')
set(ax,'ylim',ylims)
ax.YAxis(1).Color = green;

% --- plot on RHS y-axis: RMSE ----------------------------------------------
yyaxis(ax,'right')
rmse_fill= fill(ax,grid_RMSE,rmse_pdf,'r','FaceAlpha',0.50,...
                 'EdgeColor',  'none', ...
                 'DisplayName','RMSE PDF');
ylims = ylim(ax);
RMSE_mode_line = plot(ax,[RMSE_mode  RMSE_mode ], [ylims(1) ylims(2)],'--', 'Color','k','LineWidth', 1.5);
ylabel(ax,'Density (RMSE)');
set(ax,'ylim',ylims)
set(rmse_fill, 'FaceColor', blue);

title(ax,'Distributions of KL-Divergence and RMSE')
set(ax,'FontSize',10)
ax.YAxis(2).Color = blue;
legend(ax,[kl_fill,rmse_fill],'Location','best','Box','off')
set(ax,'Xtick',[0,KL_mode,RMSE_mode,1,1.5,2]);
%% Reconstructed spectrogram with minimal RMSE
gca = axs{2,2};
x_scale = (x_pick_RMSE - min(x_pick_RMSE)) ./ (max(x_pick_RMSE) - min(x_pick_RMSE));
recon = mode_wake*x_scale' + mode_sleep*(1-x_scale)';
hdl = imagesc(gca,t,w,recon);
colormap turbo;
set(gca,'ylim',[1,15]);
hdl.Parent.YDir = 'normal';
hdl.Parent.XDir = 'normal';
set(gca,'FontSize',10);
set(gca,'xlim',[t(1),t(end)]);
set(gca,'clim',[0,0.16]);
ylabel(gca,'Frequency [Hz]');
title(gca,'Model reconstruction (Minimal RMSE)');
%% Reconstructed spectrogram with minimal KL
gca = axs{2,1};
x_scale = (x_map - min(x_map)) ./ (max(x_map) - min(x_map));
recon = mode_wake*x_scale' + mode_sleep*(1-x_scale)';
hdl = imagesc(gca,t,w,recon);
colormap turbo;
set(gca,'ylim',[1,15]);
hdl.Parent.YDir = 'normal';
hdl.Parent.XDir = 'normal';
set(gca,'FontSize',10);
set(gca,'xlim',[t(1),t(end)]);
set(gca,'clim',[0,0.16]);
ylabel(gca,'Frequency [Hz]');
title(gca,'Model reconstruction (Joint MAP)');
xlabel(gca,'Time [s]');

%% SVD Embedding (ground truth)
gca = axs{2,3};
combo = mode_wake*mu_scale' + mode_sleep*(1-mu_scale)';
hdl = imagesc(gca,t,w,combo);
set(gca,'ylim',[1,15]);
set(gca,'xlim',[t(1),t(end)]);
colormap turbo;
ylabel(gca,'Frequency [Hz]');
title(gca,'SVD embedding (ground truth)');
set(gca,'FontSize',10);
hdl.Parent.YDir = 'normal';
hdl.Parent.XDir = 'normal';
set(gca,'clim',[0,0.16]);

%% Raw spectrogram
gca = axs{2,4};
ds_ratio = 10;
sm_wav = smoothdata(abs(sm_wav),2,"gaussian",300);
raw_spec = sm_wav(:,1:ds_ratio:end);
hdl = imagesc(gca,t,w,raw_spec);
set(gca,'ylim',[1,15]);
% set(gca,'XTick',[0,100,200,300,400,500]); 
set(gca,'xlim',[t(1),t(end)]);
colormap turbo;
set(gca,'clim',[-8,30]);
set(gca,'FontSize',10);
title(gca,['Raw time-frequency plot']);
ylabel(gca,'Frequency [Hz]');
hdl.Parent.YDir = 'normal';
hdl.Parent.XDir = 'normal';

saveas(gcf,'../figures/fig7_real_validation.png');

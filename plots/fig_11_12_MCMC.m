%% Compare the minimal RMSE trajectory
clc,clear all;close all;

ls = dir('../embedding/sub*.mat');
% 
MCMC_folder = '../data/Inference/MCMC_sub';
% corresponds to MCMC_sub16
fig = figure('Position',[316   131   1050  600]);
for idx = 1:length(ls)
    subplot(4,5,idx)
    load(['../embedding/',ls(idx).name]);
    MCMC_file = [MCMC_folder,num2str(idx-1),'.mat'];
    load(MCMC_file);
    MAP_traj = pp_sample(:,min_RMSE_idx+1);
    t= [0:length(ds_mu)-1]/100;hold on;
    plot(t,ds_mu);hold on;
    plot(t,MAP_traj);
    set(gca,'xlim',[0,t(end)]);
    tit = ['sub',num2str(idx)];
    set(gca,'FontSize',9);
    title(tit,'FontSize',12);
    xlabel('time [s]','FontSize',10);
    set(gca,'ylim',[-2,2.5]);
    if idx == 19
        leg = legend({'$\mu$', '$x_{min\_RMSE}(t)$'}, ...
            'Interpreter','latex','FontSize',12);
        legend boxoff;
        set(leg,'Position',[0.82 0.1 0.05 0.2]) % [x y w h] normalized
    end
end
saveas(gcf,'../figures/fig11_mu_x_RMSE.png');

%% Let's show the KL distribution and RMSE distribution
clc,close all;
fig = figure('Position',   1.0e+03 *[ 0.1130    0.0898    1.3296    0.6528
]);

blue = [0.2980	0.4471	0.6902];
green = [0.3333	0.6588	0.4078];
red = [0.7686	0.3059	0.3216];

for idx = 1:length(ls)
    subplot(4,5,idx)
    load(['../embedding/',ls(idx).name]);
    MCMC_file = [MCMC_folder,num2str(idx-1),'.mat'];
    load(MCMC_file);
    KL_grid = [0:0.01:2];
    RMSE_grid = [min(RMSE)-0.1:0.01:max(RMSE)+0.1];% 
    ax = gca;
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
    % ylims = ylim(ax);
    % map_kl = plot(ax,[KL_map KL_map], [ylims(1) ylims(2)], '-','Color',[0.3,0.3,0.3],'LineWidth', 1.5);
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

    tit = ['sub',num2str(idx)];
    set(ax,'FontSize',9);
    title(ax,tit,'FontSize',12);
    ax.YAxis(2).Color = blue;
    set(ax,'Xtick',[0,1,2])

end
saveas(gcf,'../figures/fig12_dist.png');

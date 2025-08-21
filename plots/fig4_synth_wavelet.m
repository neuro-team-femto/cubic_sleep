clc,clear all;close all;

% Load the mode from one sample subject for reconstructing spectrogram
load("mode_wake.mat");
load("mode_sleep.mat");

% Wavelet parameter
wavelet_name = 'cmor1-1.5';
fs = 100;
central_freq = centfrq(wavelet_name);
freq_start = 0.5;
freq_end = 20;
num_bin = 200;
w = linspace(freq_start,freq_end,num_bin);
scales = (central_freq/(1/fs))./w;
ds_ratio = 10;
 
% Parameter Settings
dt = 0.01;
len = 10000;
t = (0:len-1)*dt;
alpha_vals = [0.06,0.4,1];
sigma_vals = [0.5,0.8,1];
t0 = 50;
tspan = t - t0;
X0 = 0.8;

% Fixing random seed
RNG = [368,368,368];
disp(['RNG = ',num2str(RNG(1))]);

%Plot settings
figure('Position',[257.0000  242.0000  902.4000  420.0000]);
order  = {'a','b','c','d','e','f','g','h','i'};
set(gca,'clim',[-8,30]);
set(gca,'ylim',[1,15]);

for a_idx = 1:length(alpha_vals)
    alpha = alpha_vals(a_idx);
    beta = tanh(alpha*(t-t0));
    for s_idx = 1:length(sigma_vals)
        rng(RNG(s_idx));
        sigma = sigma_vals(s_idx);
        disp(['alpha  = ',num2str(alpha),' sigma = ',num2str(sigma)]);
        X = Euler_Maruyama(t, X0, @fun_cubic, @fun_brown,sigma,beta);
        X_norm = (X - min(X)) ./ (max(X)-min(X));
        X_scale = 4.5*(X+1)+1;
        mu = (X+1)/2;

        subplot(3,3,3*(a_idx-1)+s_idx);
        recon = mode_wake*mu + mode_sleep*(1 -mu) ;
        
        hdl = imagesc(gca,tspan,w, recon);hold on;
        plot(tspan,X_scale,'LineWidth',0.6,'Color',[100,100,100]/255);
        plot(tspan,-4.5*tanh(alpha * (t - t0))+5.5,'--','Color','r','LineWidth',2);
        hdl.Parent.YDir = 'normal';
        colormap turbo;
        tit = ['\sigma = ',num2str(sigma),' \alpha = ',num2str(alpha)];
        title(tit);
        label = [order{3*(a_idx-1)+s_idx},')'];
        text(0.02,0.88,label,'FontSize', 11, 'FontWeight', 'bold', 'Units', 'normalized')
        set(gca,'clim',[-0.1,0.2]);
        set(gca,'ylim',[1,15]);
        set(gca,'fontsize',12);
    end
end
saveas(gcf,'..\figures\fig4_synthe_spectrogram.png');

function X = Euler_Maruyama(time, X0, A, G,sigma,beta)
    dimension = length(X0);
    N = size(G(time(1),X0,sigma),2);
    dt = time(2)-time(1);
    % Brownian increaments
    dW = sqrt(dt)*randn(N, length(time)-1);
    % Discretized brownian paths
    X = zeros(dimension, length(time));
    X(:,1) = X0;
    Xtemp = X0;
    for i=1:length(time)-1
        Winc = dW(:,i);
        betai = beta(i);
        A1 = A(time(i), Xtemp,betai);
        G1 = G(time(i), Xtemp,sigma);
        Xtemp1 = Xtemp + A1*dt + G1*Winc;
        A2 = A(time(i+1), Xtemp1,betai);
        G2 = G(time(i+1), Xtemp1,sigma);
        Xtemp = Xtemp + 0.5*(A1+A2)*dt + (G2+G1)*0.5*Winc;
        % Xtemp = Xtemp1;
        X(:,i+1) = Xtemp;
    end
end

function G = fun_brown(time,x,sigma)
    G = [sigma];
end

function y = fun_cubic(time,x,beta)
    % rho_tar = rho_val;
    y = [-1*((x(1)+1)*(x(1)-beta)*(x(1)-1))];
end



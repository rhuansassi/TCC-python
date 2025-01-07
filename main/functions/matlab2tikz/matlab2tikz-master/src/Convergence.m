clc; clear all; close all;
scalingFactorRVNN = 0.5;
epochs = 1:70;

hfig = figure; 
fname = 'myfigure';

color1 = '#ff6666';
color2 = '#b300b3';
color3 = '#52527a';
color4 = '#00ccff';
color5 = '#339933';
color6 = '#996633';
color7 = '#00ff00';
color8 = '#e6e600';

picturewidth = 20; % set this parameter and keep it forever


% SCFNN
load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_16QAM_16Cp.mat');
plot(epochs, 10*log10(mean(mseTrain_SCF).*scalingFactorSCF), 'Color', color1, 'LineWidth', 2);
hold on;

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_16QAM_8Cp.mat');
plot(epochs, 10*log10(mean(mseTrain_SCF).*scalingFactorSCF), 'Color', color2, 'LineWidth', 2);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_64QAM_16Cp.mat');
plot(epochs, 10*log10(mean(mseTrain_SCF).*scalingFactorSCF), 'Color', color3, 'LineWidth', 2);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_64QAM_8Cp.mat');
plot(epochs, 10*log10(mean(mseTrain_SCF).*scalingFactorSCF), 'Color', color4, 'LineWidth', 2);


% RVNN
load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_16QAM_16Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(epochs,10*log10(RVNNLoss.*scalingFactorRVNN), 'Color', color5, 'LineWidth', 2);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_16QAM_8Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(epochs,10*log10(RVNNLoss.*scalingFactorRVNN), 'Color', color6, 'LineWidth', 2);


load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_64QAM_16Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(epochs,10*log10(RVNNLoss.*(scalingFactorRVNN)), 'Color', color7, 'LineWidth', 2);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_64QAM_8Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(epochs,10*log10(RVNNLoss.*(scalingFactorRVNN)), 'Color', color8, 'LineWidth', 2);

% Val SCFNN
load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_16QAM_16Cp.mat');
plot(5:5:70, 10*log10(mean(mseTrain_SCF(:,5:5:70).*scalingFactorSCF)), '*', 'Color', color1);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_16QAM_8Cp.mat');
plot(5:5:70, 10*log10(mean(mseTrain_SCF(:,5:5:70).*scalingFactorSCF)), '*', 'Color', color2);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_64QAM_16Cp.mat');
plot(5:5:70, 10*log10(mean(mseTrain_SCF(:,5:5:70).*scalingFactorSCF)), '*', 'Color', color3);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_64QAM_8Cp.mat');
plot(5:5:70, 10*log10(mean(mseTrain_SCF(:,5:5:70).*scalingFactorSCF)), '*', 'Color', color4);


% Val RVNN
load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_16QAM_16Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(5:5:70,10*log10(RVNNLoss(:,5:5:70).*scalingFactorRVNN), '*', 'Color', color5);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_16QAM_8Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(5:5:70,10*log10(RVNNLoss(:,5:5:70).*scalingFactorRVNN), '*', 'Color', color6);


load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_64QAM_16Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(5:5:70,10*log10(RVNNLoss(:,5:5:70).*scalingFactorRVNN), '*', 'Color', color7);

load('C:\Projects\tcc_v2\TCC\main\Results\Convergence\Conv_Linear_64QAM_8Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(5:5:70,10*log10(RVNNLoss(:,5:5:70).*scalingFactorRVNN), '*', 'Color', color8);

% lgd = legend('SCFNN - 16QAM 16CP', 'SCFNN - 16QAM 8CP',  'SCFNN - 64QAM 16CP',  'SCFNN - 64QAM 8CP',   ...
%               'RVNN - 16QAM 16CP', 'RVNN - 16QAM 8CP', 'RVNN - 64QAM 16CP', 'RVNN - 64QAM 8CP');
%        
grid on;

% lgd = legend('SCF - 16QAM 16CP Train', 'SCF - 16QAM 16CP Val', 'RVNN - 16QAM 16CP Train', 'RVNN - 16QAM 16CP Val', 'SCF - 16QAM 8CP Train', 'SCF - 16QAM 8CP Val', 'RVNN - 16QAM 8CP Train', 'RVNN - 16QAM 8CP Val',  ...
%              'SCF - 64QAM 16CP Train', 'SCF - 64QAM 16CP Val', 'RVNN - 64QAM 16CP Train', 'RVNN - 64AM 16CP Val', 'SCF - 64QAM 8CP Train', 'SCF - 64QAM 8CP Val', 'RVNN - 64QAM 8CP Train', 'RVNN - 64QAM 8CP Val');
%          
lgd.Location = 'northeast';
xlabel('Epoch'); ylabel('MSE');
hw_ratio = 0.65; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',13) % adjust fontsize to your document

set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
print(hfig,fname,'-dpng','-painters')




matlab2tikz('Convergence.tex')
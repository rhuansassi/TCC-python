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

Ax(1) = axes(hfig);

xlabel('Epoch'); ylabel('MSE [dB]');

hold on;

% SCFNN
load('C:\Users\marco\projects\TCC\main\Results\Convergence\Conv_Linear_16QAM_16Cp.mat');
Y1 = plot(epochs, 10*log10(mean(mseTrain_SCF).*scalingFactorSCF), 'Color', color1, 'LineWidth', 2, 'Parent', Ax(1));

load('C:\Users\marco\projects\TCC\main\Results\Convergence\Conv_Linear_16QAM_8Cp.mat');
Y2 = plot(epochs, 10*log10(mean(mseTrain_SCF).*scalingFactorSCF), 'Color', color2, 'LineWidth', 2, 'Parent', Ax(1));


% RVNN
load('C:\Users\marco\projects\TCC\main\Results\Convergence\Conv_Linear_16QAM_16Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
Y3 = plot(epochs,10*log10(RVNNLoss.*scalingFactorRVNN), 'Color', color5, 'LineWidth', 2, 'Parent', Ax(1));

load('C:\Users\marco\projects\TCC\main\Results\Convergence\Conv_Linear_16QAM_8Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
Y4 = plot(epochs,10*log10(RVNNLoss.*scalingFactorRVNN), 'Color', color6, 'LineWidth', 2, 'Parent', Ax(1));

% Val SCFNN
load('C:\Users\marco\projects\TCC\main\Results\Convergence\Conv_Linear_16QAM_16Cp.mat');
plot(5:5:70, 10*log10(mean(mseTrain_SCF(:,5:5:70)).*scalingFactorSCF), '*', 'Color', color1, 'Parent', Ax(1));

load('C:\Users\marco\projects\TCC\main\Results\Convergence\Conv_Linear_16QAM_8Cp.mat');
plot(5:5:70, 10*log10(mean(mseTrain_SCF(:, 5:5:70)).*scalingFactorSCF), '*', 'Color', color2, 'Parent', Ax(1));


% Val RVNN
load('C:\Users\marco\projects\TCC\main\Results\Convergence\Conv_Linear_16QAM_16Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(5:5:70,10*log10(RVNNLoss(:,5:5:70).*scalingFactorRVNN), '*', 'Color', color5, 'Parent', Ax(1));

load('C:\Users\marco\projects\TCC\main\Results\Convergence\Conv_Linear_16QAM_8Cp.mat');
RVNNLoss = getRVNNLoss(tr.TrainingLoss, length(epochs));
plot(5:5:70,10*log10(RVNNLoss(:,5:5:70).*scalingFactorRVNN), '*', 'Color', color6, 'Parent', Ax(1));

    
% set(Ax(1), 'Box','off')
lgd = legend(Ax(1), [Y1 Y2 Y3 Y4], 'SCFNN - 16QAM 16CP', 'SCFNN - 16QAM 8CP', ...
                                    'RVNN - 16QAM 16CP', 'RVNN - 16QAM 8CP', 'Location', 'northeast');


grid on;
hw_ratio = 0.65; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',13) % adjust fontsize to your document

set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
print(hfig,fname,'-dpng','-painters')


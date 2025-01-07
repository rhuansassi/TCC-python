clc; clear all; close all;

snrValues = 1:11;
hfig = figure; 
fname = 'myfigure';

picturewidth = 20; % set this parameter and keep it forever
load('C:\Projects\tcc_v2\TCC\main\Results\30ns\NonLinear\NonLinear_16QAM_16Cp.mat');
semilogy(EbNOdB(snrValues),LS_SER(snrValues),'Color', ["#1b998b"],'LineWidth',1.5);
hold on;
semilogy(EbNOdB(snrValues),LMMSE_SER(snrValues),'Color', ["#2d3047"],'LineWidth',1.5);
hold on;
semilogy(EbNOdB(snrValues),RVNN_SER(snrValues),'Color', ["#9fcc2e"],'LineWidth',1.5);
hold on;
semilogy(EbNOdB(snrValues),SCF_SER(snrValues),'Color', ["#347fc4"],'LineWidth',1.5);
hold on;
semilogy(EbNOdB(snrValues),LMS_SER(snrValues),'Color', ["#d55762"],'LineWidth',1.5);  
hold on;
xlabel('SNR [dB]'); ylabel('SER')
grid on;
lgd = legend('LS', 'LMMSE', 'RVNN', 'SCFNN', 'LMS');
lgd.Location = 'southwest';
 
hw_ratio = 0.65; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',13) % adjust fontsize to your document

set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
print(hfig,fname,'-dpng','-painters')





matlab2tikz('NonLinear_16QAM.tex')
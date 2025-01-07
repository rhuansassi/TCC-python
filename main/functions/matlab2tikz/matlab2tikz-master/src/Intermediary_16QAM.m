clc; clear all; close all;

snrValues = 1:11;
hfig = figure; 
fname = 'myfigure';

picturewidth = 20; % set this parameter and keep it forever

load('C:\Users\marco\projects\TCC\main\Results\30ns\Linear\Linear_16QAM_16Cp.mat');
semilogy(EbNOdB(snrValues),LS_SER(snrValues),'Color', ["#2d3047"],'LineWidth',2.0);
hold on;
semilogy(EbNOdB(snrValues),LMMSE_SER(snrValues),'Color', ["#c4253b"],'LineWidth',2.0);
hold on;
semilogy(EbNOdB(snrValues),RVNN_SER(snrValues),'Color', ["#9fcc2e"],'LineWidth',2.0);
hold on;
semilogy(EbNOdB(snrValues),SCF_SER(snrValues),'Color', ["#347fc4"],'LineWidth',2.0);
hold on;
xlabel('SNR [dB]'); ylabel('BER')
grid on;

load('C:\Users\marco\projects\TCC\main\Results\30ns\Linear\Linear_16QAM_8Cp.mat');
semilogy(EbNOdB(snrValues),LS_SER(snrValues),'--','Color', ["#2d3047"],'LineWidth',2.0);
hold on;
semilogy(EbNOdB(snrValues),LMMSE_SER(snrValues),'--','Color', ["#c4253b"],'LineWidth',2.0);
hold on;
semilogy(EbNOdB(snrValues),RVNN_SER(snrValues),'--','Color', ["#9fcc2e"],'LineWidth',2.0);
hold on;
semilogy(EbNOdB(snrValues),SCF_SER(snrValues),'--','Color', ["#347fc4"],'LineWidth',2.0);
hold on;
xlabel('SNR [dB]'); ylabel('BER')
grid on;

 
lgd = legend('LS 16CP', 'LMMSE 16CP', 'RVNN 16CP', 'SCFNN 16CP', ...
             'LS 8CP', 'LMMSE 8CP', 'RVNN 8CP', 'SCFNN 8CP');
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





matlab2tikz('IntermediaryLinear_16QAM.tex')
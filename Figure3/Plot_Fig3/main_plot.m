clc;
close all;
clear all;

figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.3010, 0.7450, 0.9930];
fs2 = 14;

load('trainable_Q.mat');
f_trainableQ = loss_list;

load('BNQ.mat');
f_BNQ = loss_list;

semilogy(epoch_list,f_trainableQ,'-.','lineWidth',2,'markersize',8);
hold on;
semilogy(epoch_list,f_BNQ,'lineWidth',2,'markersize',8);
ylim([0.005,1]);

lg=legend('Dynmic ranges learned by back-propagation training',...
          'Dynmic ranges obtained from the batch statistics',...
           'Location','northeast');
set(lg,'Fontsize',fs2-2);
set(lg,'Interpreter','latex');
grid on
xlabel('Number of Training Epochs','Interpreter','latex','FontSize',fs2);
ylabel('Average MSE','Interpreter','latex','FontSize',fs2);


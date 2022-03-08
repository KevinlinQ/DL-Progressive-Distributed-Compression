clear; close all
% figure;
figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.3010, 0.7450, 0.9930];
fs2 = 14;

load Data_performances.mat
mse_lower_bound = mse_LB_k;
mse_cd = mse_CD_Q_simul;
load Data_performances_EVD.mat
mse_evd = mse_EVD_Q_simul;
load 'data_progressive_b_m_n_bits_dim_k(3, 64, 6, 6).mat'
DNN_progressive = loss_all;
dim=[2,3,4,5,6];
DNN = nan(1,length(dim));
for ii=1:length(dim)
   f = sprintf('data_b_m_n_bits_dim_k(3, 64, 6, 6, %d).mat',dim(ii));
   load(f);
   DNN(ii) = loss_dnn_test;    
end
DNN_progressive_loss_sum = [0.019689545, 0.013289923, 0.009219772, 0.0076133544, 0.0061825393];

    

semilogy(dim,mse_evd,'v-','color',color2,'lineWidth',2,'markersize',8);
hold on
% semilogy(dim,DNN_progressive,'s-','color',color5,'lineWidth',2,'markersize',8);
% hold on
semilogy(dim,DNN_progressive_loss_sum,'*-','color',color5,'lineWidth',2,'markersize',12);
hold on
semilogy(dim,DNN,'o--','color',color4,'lineWidth',2,'markersize',8);
hold on
semilogy(dim,mse_cd,'>-','color',color3,'lineWidth',2,'markersize',8);
hold on
semilogy(dim,mse_lower_bound,'s-','color',color1,'lineWidth',2,'markersize',8);
hold on 
lg=legend('EVD using local CSI (Progressive)',...
    'DNN using local CSI (Progressive)',...
    'DNN using local CSI (Non-progressive)',...
    'BCD using global CSI (Non-progressive)','Lower bound',...
    'Location','northeast');
set(lg,'Fontsize',fs2-2);
set(lg,'Interpreter','latex');
grid on
xlabel('Dimension of Compressed Signals $(K)$','Interpreter','latex','FontSize',fs2);
ylabel('Average MSE','Interpreter','latex','FontSize',fs2);
xticks([2,3,4,5,6])
ylim([0.004,0.11])

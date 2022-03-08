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

dim_global=3;
dim_local=3.8;

M=64;
N=6;
T=100:100:1000;
transmission_num_global = dim_global*(M+1+T)+M*N;
transmission_num_local = dim_local*N+dim_local.^2+dim_local*T;

plot(T,transmission_num_local./T,'*-','color',color5,'lineWidth',2,'markersize',12);
hold on
plot(T,transmission_num_global./T,'>-','color',color3,'lineWidth',2,'markersize',8);
hold on 


lg=legend('DNN using local CSI', 'BCD using global CSI',...
    'Location','northeast');
set(lg,'Fontsize',fs2-2);
set(lg,'Interpreter','latex');
grid on
xlabel('Number of Estimations per Coherence Interval ($T$)','Interpreter','latex','FontSize',fs2);
ylabel('Normalized Communication Cost','Interpreter','latex','FontSize',fs2);

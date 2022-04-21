load('Z:\Neural Computational Design\Paper Material\Photonic spheres\results\just selection.mat')
load('Z:\Neural Computational Design\Paper Material\Photonic spheres\results\integer_inversion.mat')
load('Z:\Neural Computational Design\Paper Material\Photonic spheres\results\target_test.mat')
load('Z:\Neural Computational Design\Paper Material\Photonic spheres\results\rounding.mat')

plot_color = {'#0072BD','#D95319','#4DBEEE','#77AC30'};
k=figure;

plot(400:2:800,target_spec(1,:), '-','LineWidth',2.0,'color', plot_color{4});
hold on
plot(400:2:800,Integer_spec(1,:), '--','LineWidth',2.0,'color', plot_color{2});
hold on  
plot(400:2:800,rounded_spec(1,:), ':','LineWidth',2.0,'color', plot_color{3});
hold on    
% plot(400:2:800,target_spec(1,:), '-.','MarkerSize',10,'color', plot_color{4});
% hold on    

% title('Photonic inversion')
xlim([400,800])
ylim([3,4.5])
xlabel('Wavelength (nm)')
ylabel('\sigma / \pir^{2}')

legend(' Target',' Integer constrained', ' Rounded')
set(gca,'FontSize',12)
set(k,'Units','Inches');
pos = get(k,'Position');
set(k,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3)-0.3, pos(4)])
print(k,'Z:\Neural Computational Design\Paper Material\Photonic spheres\results\Photonic.pdf','-dpdf','-r0')

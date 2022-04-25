% run("MILP/Data/params.m")
try
    load('MILP/Data/integer_inversion.mat')
catch
    disp("The results for Nano-Photonics Integer Inversion not found. Make sure to run the corresponding experiment first")
end

try
    load('MILP/Data/rounded_inversion.mat')
catch
    disp("The results for Nano-Photonics Rounded Inversion not found. Make sure to run the experiment first")
end
addpath('MILP/') 

for i=1:17
    rounded_spec_all{i} = SimpleNeuralNet(rounded_design_all{i}/7, w_numpy);
end
h=figure;
cnt = 0;
for i=[1, 4, 6, 9, 10, 13, 14, 15]
    cnt = cnt+1;
    subplot(4,4,cnt)
    plot(400:2:800, optimal_integer_spec_all{i},'Color','[0.9290 0.6940 0.1250]','LineWidth',1.2)
    hold on
    plot(400:2:800, gt_spec_all{i},'Color','[0.4660 0.6740 0.1880]','LineWidth',1.2)
    hold on
    plot(400:2:800, rounded_spec_all{i},'--','Color','[0.1 0.2 0.8]','LineWidth',1.2)

    set(gca,'yTick',[])
    set(gca,'xTick',[])
        if ~mod(cnt+3,4)
        set(gca,'yTick',[2.8 4.2])
    end
    
ylim([2.8,4.2])
xlim([400,800])
end
for i=[2, 3, 5, 7, 8, 11, 12, 16]
    cnt = cnt+1;
    subplot(4,4,cnt)
    plot(400:2:800, optimal_integer_spec_all{i},'Color','[0.9290 0.6940 0.1250]','LineWidth',1.2)
    hold on
    plot(400:2:800, gt_spec_all{i},'Color','[0.4660 0.6740 0.1880]','LineWidth',1.2)
    hold on
    plot(400:2:800, rounded_spec_all{i},'--','Color','[0.1 0.2 0.8]','LineWidth',1.2)

    set(gca,'yTick',[])
    set(gca,'xTick',[])
    if ~mod(cnt+3,4)
        set(gca,'yTick',[0.5 5])
    end
    if cnt>=13
        set(gca,'xTick',[400 800])
    end
    
ylim([.5,5])
xlim([400,800])
end

 legend('Discrete inversion','GT','Rounded continuous inversion')
%  subplot(4,6,16)
ylabel('\sigma / \pir^{2}')
xlabel('Wavelength (nm)')

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
name_tmp='MILP/Data/nano_sphere_performance.pdf';
print(h,name_tmp,'-dpdf','-r0')
% mean(cell2mat(obj_all))
% error_optima = mean(cell2mat(obj_all))
% error_rounded =mean(cell2mat(obj_val_all))
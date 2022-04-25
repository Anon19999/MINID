clear
try
    load('Data/rounded_inversion.mat')
catch
    disp("The results for Nano-Photonics Rounded Inversion not found. Make sure to run the experiment first")
    return
end
try
    load('Data/integer_inversion.mat')
catch
    disp("The results for Nano-Photonics Integer Inversion not found. Make sure to run the corresponding experiment first")
end




h=figure;
for i=1:16
    subplot(4,4,i)
    plot(400:10:700, optimal_integer_spec_all{i},'Color','[0.9290 0.6940 0.1250]','LineWidth',1.2)
    hold on
     plot(400:10:700, gt_spec_all{i},'Color','[0.4660 0.6740 0.1880]','LineWidth',1.2)

    hold on
    plot(400:10:700, rounded_spec_all{i},'--','Color','[0.1 0.2 0.8]','LineWidth',1.2)

    set(gca,'yTick',[])
    set(gca,'xTick',[])
xlim([400,700])
ylim([0 1])
    if ~mod(i+3,4)
        set(gca,'yTick',[0 1])
    end
    if i>=13
        set(gca,'xTick',[400 700])
    end

end

ylabel('Reflectance factor')
xlabel('Wavelength (nm)')
 legend('Discrete inversion','GT','Rounded continuous inversion')


set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
name_tmp='Data/contoning_integer_inversion.pdf';
print(h,name_tmp,'-dpdf','-r0')
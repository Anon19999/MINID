 load('Data/MILP_map_approx_network.mat')
addpath('correct_color_transform')
load('../NA/Results/NA_performance.mat')
load('Data/spec_map_duotone.mat')
xyzBarIlluminants

refwhiterefl = ones(1,31); 
lightsource = D65(3:33); 

rgb_gt = sp2xyz(spec_map,lightsource,xbar(3:33),ybar(3:33),zbar(3:33)); 
rgb_gt = double(xyz2rgb(rgb_gt/100,'OutputType','uint8'))/255;

rgb_MILP = sp2xyz(MILP_map_all,lightsource,xbar(3:33),ybar(3:33),zbar(3:33)); 
rgb_MILP = double(xyz2rgb(rgb_MILP/100,'OutputType','uint8'))/255;

rgb_NA = sp2xyz(NA_performance,lightsource,xbar(3:33),ybar(3:33),zbar(3:33)); 
rgb_NA = double(xyz2rgb(rgb_NA/100,'OutputType','uint8'))/255;


h=figure('Name', 'MILP reproduced painting');
imagesc(IND)
colormap(rgb_MILP)
set(gca,'visible','off') 
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'Data/MILP_reproduced_painting.pdf','-dpdf','-r0')




h=figure('Name', 'Original Painting');
imagesc(IND)
colormap(rgb_gt)
set(gca,'visible','off')   
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'Data/Original_painting.pdf','-dpdf','-r0')


h=figure('Name', 'NA Reproduced Painting');
imagesc(IND)
colormap(rgb_NA)
set(gca,'visible','off')   
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'Data/NA_reproduced_painting.pdf','-dpdf','-r0')


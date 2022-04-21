addpath('..\..\44-ink-net\MILP\spectural transformation')
L=7;
tmp= 0.05:0.001:0.95;
% subplot(L,1,1)
h=figure;
  gray_spec_=repmat(tmp,31,1)';

[CIEXYZ, LAB_test] = Spec2LAB('D65','CIE 1931',gray_spec_);
rgb_gray = xyz2rgb(CIEXYZ/100,'OutputType','uint8');
rgb_gray=reshape(rgb_gray,[901 1 3]);
rgb_gray=repmat(rgb_gray,1,150);
rgb_gray=permute(rgb_gray,[2 1 3]);
imshow(rgb_gray);

xyzBarIlluminants

% title('Ground Truth')

set(gca,'FontSize',15)
set(h,'Units','Inches', 'Name', 'Gray Ramp GT');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'Data/GT_grayramp.pdf','-dpdf','-r0')

%%
load('Data/MILP_grayramp.mat')

h=figure;
gray_spec =  graysample_all;

[CIEXYZ, LAB_test] = Spec2LAB('D65','CIE 1931',gray_spec);
rgb_gray = xyz2rgb(CIEXYZ/100,'OutputType','uint8');
rgb_gray=reshape(rgb_gray,[901 1 3]);
rgb_gray=repmat(rgb_gray,1,150);
rgb_gray=permute(rgb_gray,[2 1 3]);
imshow(rgb_gray);


 set(gca,'FontSize',15)   
set(h,'Units','Inches', 'Name', 'MILP Gray 4-Ink');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'Data/MILP_gray_4ink.pdf','-dpdf','-r0')

clear 
clc

[file,path] = uigetfile('E:\Navid\CAM\Matlab\Data_set\test_LAB\full_network\*.npy' );
filename = [path file];
spec1 = double(readNPY(filename));

% load('E:\Navid\CAM\Matlab\Data_set\watercolor_dataset\spec_watercolor.mat')
% spec1=spec;
[file,path] = uigetfile('E:\Navid\CAM\Matlab\Data_set\test_LAB\full_network\*.npy' );
filename = [path file];
spec2= double(readNPY(filename));

mean_RMS=100*mean(vecnorm(spec1-spec2,2,2))/sqrt(31)
median_RMS=100*median(vecnorm(spec1-spec2,2,2))/sqrt(31)
max_RMS=100*max(vecnorm(spec1-spec2,2,2))/sqrt(31)
std_RMS=100*std(vecnorm(spec1-spec2,2,2))/sqrt(31)

addpath('./Useful_color_conversion_functions/') 
xyzBarIlluminants

refwhiterefl = ones(1,31); 
lightsource = TL84(3:33); 

xyzOut = sp2xyz(spec2,lightsource,xbar(3:33),ybar(3:33),zbar(3:33)); 
xyzTrans = sp2xyz(spec1,lightsource,xbar(3:33),ybar(3:33),zbar(3:33)); 

refWhiteXyz = sp2xyz(refwhiterefl,lightsource,xbar(3:33),ybar(3:33),zbar(3:33));

labOut = xyz2lab(xyzOut,refWhiteXyz); 
labTrans = xyz2lab(xyzTrans,refWhiteXyz); 

sl = 1; sc = 1; sh = 1; 

[de,~,~,~] = cie00de(labOut,labTrans,sl,sc,sh); 

meanDE = mean(de)
medianDE = median(de)
maxDE = max(de)
sdDE = std(de)
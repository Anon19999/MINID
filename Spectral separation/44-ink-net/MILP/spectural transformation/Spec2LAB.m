
function [CIEXYZ, LAB] = Spec2LAB(lightSourceTag,CIETag,spec)
%% Import the data
% lightSourceTag=input('choose your light source','s');
% CIETag=input('choose your CIE model','s');


% [file,path] = uigetfile('*.xlsx','upload the light source and the color matching functions' );
file='StandardsCIE.xlsx';
% path='Z:\thesis\Navid\CAM\Mixed Integer Ink Selection\exercise\';
[~, ~, raw] =xlsread(file,'10 nm');


raw = raw(:,1:42);
stringVectors = string(raw(:,1));
stringVectors(ismissing(stringVectors)) = '';
raw = raw(:,2:end);





%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
StandardsCIE = table;

%% Allocate imported array to column variable names
StandardsCIE.tag = stringVectors(:,1);
StandardsCIE.Var = data(:,3:end-8);

%% Clear temporary variables
clearvars data raw stringVectors;


%% Clear temporary variables
clearvars data raw stringVectors;
CIEmodel=["CIE 1964";"CIE 1931"];
index=[2:4;5:7];
T=table(CIEmodel,index);

%% CIECalculation
%add perfect white for calculation of lab. Perfect white should be measured seperatly for each measuring session 
% spec=[spec;ones(1,length(spec(1,:)))];
% white_painting=[19.17	18.21	19.75	23.97	34.29	61.50	94.29	110.35	113.65	107.33	98.79	94.29	91.30	87.81	85.65	83.16	80.34	78.12	76.94	76.04	74.84	74.20	74.57	75.36	75.84	75.93	76.13	76.86	77.93	79.07	79.69	79.85	79.62	79.38	79.40	79.59	79.68	79.85	79.80	79.74];
% white_painting=white_painting(:,5:end-5)/100;
%white paper
%  white_patches=[60.69	61.93	64.13	66.56	68.82	70.72	72.55	74.29	75.74	77.00	78.31	79.50	80.46	81.46	82.35	83.11	83.81	84.52	85.17	85.76	86.11	86.64	87.02	87.38	87.69	87.87	88.04	88.19	88.36	88.58	88.71	88.83	88.92	88.99	88.98	89.05	89.04	89.07	88.95	88.72];	
 white_patches=ones(1,40);
%film_transmittance_white
% white_patches=[88.93	88.65	88.24	87.69	87.58	87.49	87.54	87.56	87.59	87.70	87.84	87.89	87.96	88.00	88.03	88.11	88.23	88.28	88.31	88.32	88.40	88.44	88.50	88.56	88.58	88.55	88.60	88.64	88.73	88.80	88.81	88.80	88.84	88.93	88.97	89.03	89.04	89.01	89.04	88.98];
white_patches=white_patches(:,5:end-5)/100;



spec=[spec;white_patches];	


lightsource=StandardsCIE.Var(StandardsCIE.tag==lightSourceTag,1:min([length(StandardsCIE.Var(1,:)) length(spec(1,:))]));

CIE=StandardsCIE.Var(T.index(T.CIEmodel==CIETag,:),1:min([length(StandardsCIE.Var(1,:)) length(spec(1,:))]));
% CIEXYZ2=(CIE*(obj.*repmat(lightsource,size(obj,1),1))')*100/(CIE(2,:)*lightsource');
CIEXYZ=sp2xyz(spec,lightsource,CIE(1,:),CIE(2,:),CIE(3,:));
%the last one is perfect white 
CIEXYZ2=CIEXYZ';
CIEXYZ=CIEXYZ(1:end-1,:);

% scatter3(CIEXYZ2(1,1:end-1),CIEXYZ2(2,1:end-1),CIEXYZ2(3,1:end-1))
% title('CIEXYZ')
% K=CIE(2,:)*lightsource';
% CIEXYZ=CIEXYZ/K
%LAB calculation
syms tt f(tt)
f(tt)=piecewise(tt>(6/29)^3, tt^(1/3), tt<=(6/29)^3, (tt/((3*(6/29)^2)))+(4/29));
xyzWhite=CIEXYZ2(:,end);
Lstar=double(116*f(CIEXYZ2(2,1:end-1)./xyzWhite(2))-16);
astar=double(500*(f(CIEXYZ2(1,1:end-1)./xyzWhite(1))-f(CIEXYZ2(2,1:end-1)./xyzWhite(2))));
bstar=double(200*(f(CIEXYZ2(2,1:end-1)./xyzWhite(2))-f(CIEXYZ2(3,1:end-1)./xyzWhite(3))));
LAB=[Lstar' astar' bstar'];
clearvars Lstar astar bstar xyzWhite CIEXYZ2

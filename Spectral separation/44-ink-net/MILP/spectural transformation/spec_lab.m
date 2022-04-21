

%% Import the data
lightSourceTag=input('choose your light source','s');
CIETag=input('choose your CIE model','s');


% [file,path] = uigetfile('*.xlsx','upload the light source and the color matching functions' );
file='StandardsCIE.xslx';
path='C:\Users\nansari\Desktop\Navid\CAM\exercise\';
[~, ~, raw] =xlsread([path file],'10 nm');


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

% 
% %% Import the data
% 
% [file,path] = uigetfile('*.xlsx','upload the Reflectance property of the objects' );
% [~, ~, raw] =xlsread([path file],'Feuil1');
% 
% stringVectors = string(raw(:,1));
% stringVectors(ismissing(stringVectors)) = '';
% raw = raw(:,2:end);
% 
% %% Create output variable
% data = reshape([raw{:}],size(raw));
% 
% %% Create table
% objects = table;
% 
% %% Allocate imported array to column variable names
% objects.name = stringVectors(:,1);
% objects.Var = data(:,1:end);
% objects = table;
% objects.Var(i) = data(:,1:end);
%% Clear temporary variables
clearvars data raw stringVectors;
CIEmodel=["CIE 1964";"CIE 1931"];
index=[2:4;5:7];
T=table(CIEmodel,index);

%% CIECalculation
%add perfect white for calculation of lab
% obj=[obj(:,5:end-5);ones(1,length(obj(1,5:end-5)))];

lightsource=StandardsCIE.Var(StandardsCIE.tag==lightSourceTag,1:min([length(StandardsCIE.Var(1,:)) length(obj(1,:))]));

CIE=StandardsCIE.Var(T.index(T.CIEmodel==CIETag,:),1:min([length(StandardsCIE.Var(1,:)) length(obj(1,:))]));
% CIEXYZ2=(CIE*(obj.*repmat(lightsource,size(obj,1),1))')*100/(CIE(2,:)*lightsource');
CIEXYZ=sp2xyz(obj,lightsource,CIE(1,:),CIE(2,:),CIE(3,:));
CIEXYZ2=CIEXYZ';
scatter3(CIEXYZ2(1,1:end-1),CIEXYZ2(2,1:end-1),CIEXYZ2(3,1:end-1))
title('CIEXYZ')
% K=CIE(2,:)*lightsource';
% CIEXYZ=CIEXYZ/K
%LAB calculation
syms t f(t)
f(t)=piecewise(t>(6/29)^3, t^(1/3), t<=(6/29)^3, (t/((3*(6/29)^2)))+(4/29));
xyzWhite=CIEXYZ2(:,end);
Lstar=double(116*f(CIEXYZ2(2,1:end-1)./xyzWhite(2))-16);
astar=double(500*(f(CIEXYZ2(1,1:end-1)./xyzWhite(1))-f(CIEXYZ2(2,1:end-1)./xyzWhite(2))));
bstar=double(200*(f(CIEXYZ2(2,1:end-1)./xyzWhite(2))-f(CIEXYZ2(3,1:end-1)./xyzWhite(3))));
LAB=[Lstar' astar' bstar'];
figure
scatter3(Lstar,astar,bstar)
title('CIELAB')
% sqrt((Lstar(1)-Lstar(2))^2+(astar(1)-astar(2))^2+(bstar(1)-bstar(2))^2)
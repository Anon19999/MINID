function [ DeltaE94 ] = twoLab2De94( LABpatchA1,LABpatchA2 )
DeltaL=LABpatchA1(1)-LABpatchA2(1);
aA1=LABpatchA1(2);aA2=LABpatchA2(2);bA1=LABpatchA1(3);bA2=LABpatchA2(3);
CA1=(aA1^2+bA1^2)^0.5;
CA2=(aA2^2+bA2^2)^0.5;
DeltaC=CA1-CA2;
%DeltaH=(2*CA1*CA2-2*aA1*aA2-2*bA1*bA2)^0.5; % may yield complex numbers
DeltaA=aA1-aA2; DeltaB=bA1-bA2;
DeltaH=(DeltaA^2+DeltaB^2-DeltaC^2)^0.5; 

Cab=(CA1*CA2)^0.5;

Sc94=1+0.045*Cab;
Sh94=1+0.015*Cab;
Sl94=1.0;
Kl=1;Kc=1;Kh=1;

DeltaE94=((DeltaL/(Kl*Sl94))^2+(DeltaC/(Kc*Sc94))^2+(DeltaH/(Kh*Sh94))^2)^0.5;
DeltaE94=real(DeltaE94);   % Ignore imaginary part of result [phd, 16/8/2005]


end


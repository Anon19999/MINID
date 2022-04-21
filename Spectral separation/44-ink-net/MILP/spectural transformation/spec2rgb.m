spec=spec_all(Selected_patches,:);
[CIEXYZ, LAB_test] = Spec2LAB('D65','CIE 1931',spec);
rgb = xyz2rgb(CIEXYZ/100,'OutputType','uint8');
%%

%     p=ones(28900,700,3);
    for i=1:24
    temp(1:100,1:100,1)=uint8(ones(100)).*rgb(i,1);
    temp(1:100,1:100,2)=uint8(ones(100)).*rgb(i,2);
    temp(1:100,1:100,3)=uint8(ones(100)).*rgb(i,3);
    p(1+floor((i-1)/6)*100:1+floor((i-1)/6)*100+99,1+(i-1-floor((i-1)/6)*6)*100:1+(i-1-floor((i-1)/6)*6)*100+99,1:3)=temp;
    position=[1+(i-1-floor((i-1)/6)*6)*100+50 1+floor((i-1)/6)*100+50];
    p = insertText(p,position,sprintf('%d',i),'FontSize',18,'BoxColor',...
    'white','BoxOpacity',0.4,'TextColor','black');
% imwrite(temp,sprintf('E:\\Navid\\CAM\\Paper_material\\image\\Ink_library\\rgb%d.png',i))

    end
imshow(p)
% imwrite(p,'C:\Users\nansari\Desktop\Navid\CAM\reports\spectral reconstruction\rgb.jpg')
% %%
% wl=400:10:700;
% for i=1:20
%     plot(wl,reflectance(i,:))
%     hold on
%     set(gca,'YTick',[])
%     set(gca,'xTick',[])
%     xlabel('Wavelength','FontSize',15,'FontWeight','bold')
%     ylabel('T','FontSize',15,'FontWeight','bold')
% end


load('Data/two_samples_robustness.mat')
load('Data/backprob_loss_NA.mat')
for i =1:2
 
h=figure;   
% plot(double(best_loss(50*(i-1)+1:50*(i-1)+20)),'Marker','.','MarkerSize',15,'color','[0.3 0.8 0.2]','LineWidth',2)
bubblechart(1:20,double(best_loss(50*(i-1)+1:50*(i-1)+20)),devi(i,:));
blgd = bubblelegend('Max deviation');
blgd.Location = 'northwest';
blgd.NumBubbles = 3;
hold on
scatter(1:20, double(best_loss(50*(i-1)+1:50*(i-1)+20)),'.', 'red')

% plot(devi(i,:),'Marker','.','MarkerSize',15,'color','[0.8 0.3 0.6]','LineWidth',2)

% legend('solution loss','solution robustness')
xlabel('Solution ID','FontSize',20)
ylabel('Loss','FontSize',20)
% xlim([0 21])
% ylim([4 9])
if i==1
 % xlim([0 21])
ylim([0 15])
end
if i==2
  ylim([32 50])
end


set(gca,'FontSize',12)
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3)-0.15, pos(4)])
print(h,sprintf('Data/robustness_test_%d.pdf',i),'-dpdf','-r0')


end
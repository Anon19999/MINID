
%%

load('Data/Mesh_points.mat')
load('Data/soft_robot_w_b_128_128.mat')
load('Data/Performances_all.mat')
load('Data/obstacles_1_100_50000.mat')
load('Data/softrobot_withObstacle_zeta_0.2.mat')
load('Data/triangles.mat')
load('Data/targets.mat')




vertex_center = 2:3:63;
vertex_all = 2:63;
vertex_edge = setdiff(vertex_all,vertex_center);
% sample_num = 1:100:50000;

% x{1} = design_all;
for test_num =50:60
    h = figure;
%  subplot(2,2,test_num);
% test_num = 150;
rng(5)
rendom_samples = randperm(50000,1000);
n_cnt = rendom_samples(test_num);
state_params = Performances_all;
target = state_params(n_cnt,:);
target = target(:,123:124);

% target_reproduced = double(target_reproduced);
% performance_state = SimpleNeuralNet(double(x{1}), w_numpy);
% test_design =design;
performance_state = SimpleNeuralNet(design_all(test_num,:), w_numpy);
% performance_state = SimpleNeuralNet(test_design, w_numpy);


sample = reshape(performance_state,2,103);
sample = sample';
Meshpoint = reshape(Mesh_points, 2,103);
Meshpoint = Meshpoint';
finalpose = sample + Meshpoint;
% plot(finalpose(:,1),finalpose(:,2),'*')
finalpose = double(finalpose);
triplot(double(triangles)+1,finalpose(:,1),finalpose(:,2),'k')
hold on
% plot(target_reproduced(:,1)+0.25,target_reproduced(:,2)+10,'ko')
% hold on
r = 0.9;
pgon = polyshape([obstacles(n_cnt,1)-r*sqrt(2) obstacles(n_cnt,1) obstacles(n_cnt,1)+r*sqrt(2) obstacles(n_cnt,1)]...
    ,[obstacles(n_cnt,2) obstacles(n_cnt,2)+r*sqrt(2) obstacles(n_cnt,2) obstacles(n_cnt,2)-r*sqrt(2)]);
plot(pgon)
hold on
scatter(target(:,1)+0.25,target(:,2)+10,'bo')
hold on
viscircles(obstacles(n_cnt,:),0.9)
legend('MILP reproduced softrobot','obstacle', 'target')
axis equal
xlim([-6 7])
ylim([0 11])
title(sprintf('%d',test_num))
end

% h=figure;
% set(gca,'FontSize',20)
% coreset = [1, 2, 3, 5, 7, 10, 15, 20];
% plot(coreset, time_vec,'Marker','.','MarkerSize',15,'color','[0.5 0.5 0.5]','LineWidth',2)
% hold on
% 
% xlabel('Coreset size','FontSize',20)
% ylabel('Optimization time (s)','FontSize',20)
% xlim([0 24])
% ylim([0 300])
% 

set(gca,'FontSize',20)
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3)-0.15, pos(4)])
print(h,'Data\soft_robot_teaser.pdf','-dpdf','-r0')

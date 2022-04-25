load('../NA/backprob_loss_NA_150_4000samples.mat')
load('../NA/backprob_time_NA_150_4000samples.mat')
%%
load('Data/sample_101_4lay_150_log.mat')


loss_min =inf;
time_loss = [];
loss_min_all = [];
for i=1:4000
    if best_loss(i)<=loss_min
        loss_min = best_loss(i);
        loss_min_all = [loss_min_all, loss_min];
        time_loss = [time_loss, backprob_time(i)];
    end
end
h=figure;
set(gca,'FontSize',20)
semilogy(time_loss(1:end),loss_min_all(1:end),'Marker','.','MarkerSize',15,'color','[0.9290 0.6940 0.1250]','LineWidth',2)
hold on
semilogy(time_all(1:end),upperBound(1:end),'Marker','.','MarkerSize',15,'color','[0.1 0.2 0.8]','LineWidth',2)
hold on

semilogy(time_all,lowerBound,'Marker','.','MarkerSize',15,'color','[0.4660 0.6740 0.1880]','LineWidth',2)

xlim([0,1000])
legend('NA','upper bound','lowerbound')

ylabel('Loss','FontSize',20)
xlabel('Optimization time (s)','FontSize',20)

set(gca,'FontSize',20)
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3)-0.15, pos(4)])
print(h,'Data/sample_101_NA_MILP.pdf','-dpdf','-r0')
%%
load('Data/sample_201_4lay_150_log.mat')


loss_min =inf;
time_loss = [];
loss_min_all = [];
for i=4001:8000
    if best_loss(i)<=loss_min
        loss_min = best_loss(i);
        loss_min_all = [loss_min_all, loss_min];
        time_loss = [time_loss, backprob_time(i)];
    end
end
h=figure;
set(gca,'FontSize',20)
semilogy(time_loss(1:end),loss_min_all(1:end),'Marker','.','MarkerSize',15,'color','[0.9290 0.6940 0.1250]','LineWidth',2)
hold on
semilogy(time_all(1:end),upperBound(1:end),'Marker','.','MarkerSize',15,'color','[0.1 0.2 0.8]','LineWidth',2)
hold on

semilogy(time_all,lowerBound,'Marker','.','MarkerSize',15,'color','[0.4660 0.6740 0.1880]','LineWidth',2)

xlim([0,3000])
legend('NA','upper bound','lowerbound')

ylabel('Loss','FontSize',20)
xlabel('Optimization time (s)','FontSize',20)

set(gca,'FontSize',20)
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3)-0.15, pos(4)])
print(h,'Data/sample_201_NA_MILP.pdf','-dpdf','-r0')
%%
load('Data/sample_301_4lay_150_log.mat')


loss_min =inf;
time_loss = [];
loss_min_all = [];
for i=8001:12000
    if best_loss(i)<=loss_min
        loss_min = best_loss(i);
        loss_min_all = [loss_min_all, loss_min];
        time_loss = [time_loss, backprob_time(i)];
    end
end
loss_min_all(length(loss_min_all)+1) = loss_min_all(end);
time_loss(length(loss_min_all)) = 4500;
h=figure;
set(gca,'FontSize',20)
semilogy(time_loss(1:end),loss_min_all(1:end),'Marker','.','MarkerSize',15,'color','[0.9290 0.6940 0.1250]','LineWidth',2)
hold on
semilogy(time_all(1:end),upperBound(1:end),'Marker','.','MarkerSize',15,'color','[0.1 0.2 0.8]','LineWidth',2)
hold on

semilogy(time_all,lowerBound,'Marker','.','MarkerSize',15,'color','[0.4660 0.6740 0.1880]','LineWidth',2)

xlim([0,4250])
legend('NA','upper bound','lowerbound')

ylabel('Loss','FontSize',20)
xlabel('Optimization time (s)','FontSize',20)

set(gca,'FontSize',20)
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3)-0.15, pos(4)])
print(h,'Data/sample_301_NA_MILP.pdf','-dpdf','-r0')
%%
load('Data/sample_401_4lay_150_log.mat')


loss_min =inf;
time_loss = [];
loss_min_all = [];
for i=12001:13000
    if best_loss(i)<=loss_min
        loss_min = best_loss(i);
        loss_min_all = [loss_min_all, loss_min];
        time_loss = [time_loss, backprob_time(i)];
    end
end
h=figure;
set(gca,'FontSize',20)
semilogy(time_loss(1:end),loss_min_all(1:end),'Marker','.','MarkerSize',15,'color','[0.9290 0.6940 0.1250]','LineWidth',2)
hold on
semilogy(time_all(1:end),upperBound(1:end),'Marker','.','MarkerSize',15,'color','[0.1 0.2 0.8]','LineWidth',2)
hold on

semilogy(time_all,lowerBound,'Marker','.','MarkerSize',15,'color','[0.4660 0.6740 0.1880]','LineWidth',2)

xlim([0,1300])
legend('NA','upper bound','lowerbound')

ylabel('Loss','FontSize',20)
xlabel('Optimization time (s)','FontSize',20)

set(gca,'FontSize',20)
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3)-0.15, pos(4)])
print(h,'Data/sample_401_NA_MILP.pdf','-dpdf','-r0')



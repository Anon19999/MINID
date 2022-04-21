sample1_2 = load('robustness_results.mat');
devi = zeros(2,20);
all_time = zeros(2,20);

dev_tmp = cell2mat(sample1_2.deviation_all);
time_tmp = cell2mat(sample1_2.time_all);

devi(1,:) = dev_tmp(1:20);
all_time(1,:) = time_tmp(1:20);
devi(2,:) = dev_tmp(51:70);
all_time(2,:) = time_tmp(51:70);


save('two_samples_robustness.mat', 'devi', 'all_time'  )


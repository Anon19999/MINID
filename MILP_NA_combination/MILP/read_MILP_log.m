fid=fopen('Z:\Neural Computational Design\NA and MILP\ink\MILP\sample_301_4lay_150.log');
C = textscan(fid,'%s',700,'delimiter','\n', 'headerlines',27);
C{1}
fclose(fid)
for i=1:700
    time_all(i) = str2double(C{1}{i}(end-4:end-1));
    gap(i) = str2double(C{1}{i}(end-17:end-13));
        lowerBound(i) = str2double(C{1}{i}(end-26:end-18));
        upperBound(i) = str2double(C{1}{i}(end-39:end-28));      

end
save('sample_301_4lay_150_log.mat','time_all','gap','lowerBound','upperBound' )
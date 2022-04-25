logPathTemplate = 'Data/paper_result/sample_%i_4lay_150.log';
matPathTemplate = 'Data/sample_%i_4lay_150_log.mat';
for j = 101:100:401
    fid=fopen(sprintf(logPathTemplate, j));
    C = textscan(fid,'%s',1000,'delimiter','\n', 'headerlines',23);
    fclose(fid);
    for i=1:size(C{1})-21
        time_all(i) = str2double(C{1}{i}(end-4:end-1));
        gap(i) = str2double(C{1}{i}(end-17:end-13));
            lowerBound(i) = str2double(C{1}{i}(end-26:end-18));
            upperBound(i) = str2double(C{1}{i}(end-39:end-28));      

    end
    save(sprintf(matPathTemplate,j),'time_all','gap','lowerBound','upperBound' )
end
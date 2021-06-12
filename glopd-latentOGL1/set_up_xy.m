function [X, y] = set_up_xy(datasetName, fileType, dbDir, to_dense)
    if fileType == 'txt'
        filepath = [dbDir, '/', datasetName, '.', fileType];
    elseif fileType == 'bz2'
        if ~exist(datasetName, 'file')
            cmd = sprintf('bzip2 -dk %s', [dbDir, '/', datasetName, '.', fileType]);
            system(cmd)
        end
        filepath = [dbDir,'/', datasetName];
    else
        error('Invalid fileType!\n');
    end
    [y, X]  = libsvmread(filepath);
    if to_dense
        X = full(X);
    end
end

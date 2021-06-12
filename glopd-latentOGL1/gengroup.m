function [blocks,numgrp] = gengroup(dim, numgrp, grpsize)
    if grpsize * numgrp <= dim
        error("grpsize is too small to have overlapping.");
    end
    if grpsize >= dim
        error("grp_size is too large that each group has all variables.");
    end
    exceed = numgrp * grpsize - dim;
    overlap_per_group = fix(exceed / (numgrp - 1));
    blocks = {numgrp};
    for i=1:numgrp
        if i==1
            start_ = 1;
            end_  = start_+grpsize-1;
        else
            start_ = end_ - overlap_per_group + 1;
            end_ = min(start_ + grpsize -1, dim);
            if (start_ == blocks{i-1}(1)) & (end_ == blocks{i-1}(end))
                numgrp = i - 1;
                blocks = blocks{1:numgrp};
                fprintf('actual number of group %d', numgrp);
                return 
            end
        end
        blocks{i}=start_:end_;
    end
end
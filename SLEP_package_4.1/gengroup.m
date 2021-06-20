% function [blocks,numgrp] = gengroup(dim, numgrp, grpsize)
%     if grpsize * numgrp <= dim
%         error("grpsize is too small to have overlapping.");
%     end
%     if grpsize >= dim
%         error("grp_size is too large that each group has all variables.");
%     end
%     exceed = numgrp * grpsize - dim;
%     overlap_per_group = fix(exceed / (numgrp - 1));
%     blocks = {numgrp};
%     for i=1:numgrp
%         if i==1
%             start_ = 1;
%             end_  = start_+grpsize-1;
%         else
%             start_ = end_ - overlap_per_group + 1;
%             end_ = min(start_ + grpsize -1, dim);
%             if (start_ == blocks{i-1}(1)) & (end_ == blocks{i-1}(end))
%                 numgrp = i - 1;
%                 blocks = blocks{1:numgrp};
%                 fprintf('actual number of group %d', numgrp);
%                 return 
%             end
%         end
%         blocks{i}=start_:end_;
%     end
% end
function [G,W] = gengroup(dim, grpsize, overlap_ratio)
    if grpsize >= dim
        error("grp_size is too large that each group has all variables.")
    end
    overlap = fix(grpsize * overlap_ratio);
    if overlap < 1
        msg = "current config of grp_size and overlap_ratio cannot produce overlapping groups. overlap_ratio is adjusted to have at least one overlap.";
        warning(msg);
        overlap = 1;
    end
    G = [];
    w1 = [];
    w2 = [];
    w3 = [];
    start_ = 0;
    end_ = grpsize- 1;
    while (1)
        G = [G, start_:end_];
        w1 = [w1, start_];
        w2 = [w2, end_];
        w3 = [w3, sqrt(end_-start_+1)];
        start_ = end_ - (overlap - 1);
        end_ = min(start_ + grpsize - 1, dim - 1);
        if end_ == w2(end)
            break
        end
    end
    W = [w1;w2;w3];
end
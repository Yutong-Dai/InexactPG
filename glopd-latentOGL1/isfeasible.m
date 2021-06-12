function flag = isfeasible(z, alpha, blocks, weights)
    B = length(blocks);
    flag = true;
    for i=1:B
        zg = z(blocks{i});
        norm_zg = norm(zg);
        if norm_zg - alpha * weights(i) > 1e-8
            fprintf('i: %d | norm_zg %d  | radius_g %d\n',  i, norm_zg, alpha * weights(i));
            fprintf('diff:%d\n',  norm_zg - alpha * weights(i));
            flag=false;
            return
        end
    end
    return
end
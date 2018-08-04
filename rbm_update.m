function [M, b, c] = rbm_update(M, b, c, deltaM, deltab, deltac, eta)
    %% WEIGHTS UPDATE
    % weights update
    M = M + eta * deltaM;
    b = b + eta * deltab;
    c = c + eta * deltac;

end


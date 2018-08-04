function [H] = rbm_encode(X, M, b, c)
    Nh = size(c, 1);
    Ni = size(b, 1);
    Nd = size(X, 1);
    sigmoid = @(a) 1.0 ./ (1.0 + exp(-a));
    
    H = zeros(Nd, Nh);
    
    % collect encodings
    for i=1:Nd
        v0 = X(i,:)'; % > rand(Ni,1);
        h0 = sigmoid(M' * v0 + c); % > rand(Nh,1);
        
        H(i, :) = h0;
    end
end


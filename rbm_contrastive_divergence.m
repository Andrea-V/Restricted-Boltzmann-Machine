function [h0, v0, vk, hk] = rbm_contrastive_divergence(M, b, c, cd_k, x)
    Nh = size(c, 1);
    Ni = size(b, 1);
    sigmoid = @(a) 1.0 ./ (1.0 + exp(-a));
    
    % clamp training vector to visible units
    v0 = x; % > rand(Ni,1);
    
    % update hidden units
    p_h0v0 = sigmoid(M' * v0 + c);
    h0 = p_h0v0 > rand(Nh,1);

    vk = v0;
    hk = h0;
    for k = 1:cd_k
        % update visible units to get reconstruction
        p_vkhk = sigmoid(M * hk + b);
        vk = p_vkhk; % > rand(Ni, 1);

        % update hidden units again
        p_hkvk = sigmoid(M' * vk + c);
        hk = p_hkvk > rand(Nh,1);
    end
    
    % when computing gradient, we can use probabilities
    % for hidden units as well (beacuse Hinton says so)
    hk = p_hkvk;
    h0 = p_h0v0;
end


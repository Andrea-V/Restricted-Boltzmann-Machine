function [M, b, c] = rbm_init(ninput, nhidden)
    % weight matrix
    M = 0.01 * (randn(ninput, nhidden) - 0.5);
    % biases
    c = zeros(nhidden ,1);
    b = zeros(ninput, 1);
end


function [M , b, c , errors] = rbm_train(X, M, b, c, cd_k, eta, alpha, lambda, max_epochs)
    % %% INIT VARIABLES
    e = 0;
    errors = [Inf];

    Nh = size(c, 1);
    Ni = size(b, 1);
    Nd = size(X, 1);

    % %% EPOCH
    while true
        deltaM = zeros(Ni, Nh);
        deltab = zeros(Ni, 1);
        deltac = zeros(Nh ,1);
        error = 0;
        
        %shuffle inputs
        fprintf('-- shuffling inputs\n');
        X = X(randperm(size(X, 1)),:);
        
        % stochastic updates
        fprintf('-- training...\n');
        for i = 1:Nd
            % k step contrastive divergence
            [h0, v0, vk, hk] = rbm_contrastive_divergence(M, b, c, cd_k, X(i,:)');
            
            % compute gradient
            [Mgrad, bgrad, cgrad] = rbm_gradient(v0, h0, vk, hk);
            
            % momentum
            deltaM = alpha * deltaM + (1-alpha) * Mgrad;
            deltab = alpha * deltab + (1-alpha) * bgrad;
            deltac = alpha * deltac + (1-alpha) * cgrad;
            
            % weights update
            M = M + eta * deltaM;
            b = b + eta * deltab;
            c = c + eta * deltac;

            % weight decay
            M = M - lambda * M;
            
            % error metric
            error = error + norm(X(i,:)' - vk);
        end
        
        % mean error over tr samples
        errors(end + 1) = error / Nd; 

        fprintf('- epoch %d, error: %f\n', e, errors(end));
        if e > max_epochs
            break
        end
        
        e = e + 1;
    end
end


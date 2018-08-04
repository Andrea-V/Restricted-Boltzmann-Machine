function [Mgrad, bgrad, cgrad] = rbm_gradient(v0, h0, vk, hk)
    Mgrad = (v0 * h0') - (vk * hk');
    bgrad = (v0 - vk);
    cgrad = (h0 - hk);
end


function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

f=@(h) 1 ./ (1 + exp(-h)); % 1 / (1 + exp(-h));


g = f(z);



% =============================================================

end

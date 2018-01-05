function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));



predictions =  sigmoid(X*theta);

leftPart = -y' * log(predictions);

rightPart = (1 - y') * log(1 - predictions);

J = (1 / m) * (leftPart - rightPart);

grad = (1 / m) * ((predictions - y)' * X);




% =============================================================

end

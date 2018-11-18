function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

theta_reg = theta(2:size(theta));
theta_reg = [0; theta_reg];

h = X * theta;
difference = h .- y;
first_term = (1 / (2 * m)) * sum(difference .^ 2);
second_term = (lambda / (2 * m)) * sum(theta_reg .^ 2);
J = first_term + second_term;

for j = 1:size(grad)
    Xj = X(:,j);
    grad(j) = (1/m)*sum(difference .* Xj) + (lambda/m)*theta_reg(j);
endfor

% =========================================================================

grad = grad(:);

end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J=-1/(m)*(log(sigmoid(X*theta))'*y+log(1-sigmoid(X*theta))'*(1-y))+lambda/2/m*(theta(2:size(theta,1))'*theta(2:size(theta,1)));


a=(1/m*X'*(sigmoid(X*theta)-y));
grad(1)=a(1);
b=lambda*theta;
grad(2:size(theta,1))=a(2:size(theta,1))+b(2:size(theta,1))/m ;



% =============================================================

end

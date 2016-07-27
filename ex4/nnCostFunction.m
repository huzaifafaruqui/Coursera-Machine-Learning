function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
%K=num_labels;
yd = eye(num_labels);
yd = yd(y,:);

a1=[ones(m,1),X];  %5000x401
z2=Theta1*a1'; %25x5000
a2=sigmoid(z2);
a2=[ones(1,m);a2]; %26x5000
z3=Theta2*a2;
hx=sigmoid(z3)'; %5000x10
%y=5000x10

b=((-yd).*log(hx)-(1-yd).*log(1-hx));
J=1/m.*sum(sum(b));

%dont regularise first column of thetas
rTheta1=Theta1(:,2:end); %25x400
rTheta2=Theta2(:,2:end); %10x25
J=J+(lambda/(2*m)).*(sum(sum(rTheta1.^2))+sum(sum(rTheta2.^2)));

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%theta2-10x26
X = [ones(m,1) X];

delta_accum_1 = zeros(size(Theta1));
delta_accum_2 = zeros(size(Theta2));

for t = 1:m
	a_1 = X(t,:);  
	z_2 = a_1 * Theta1';
	a_2 = [1 sigmoid(z_2)];
	z_3 = a_2 * Theta2';
	a_3 = sigmoid(z_3);
	y_i = zeros(1,num_labels);
	y_i(y(t)) = 1;
	
	delta_3 = a_3 - y_i;
	delta_2 = delta_3 * Theta2 .* sigmoidGradient([1 z_2]);
	
	delta_accum_1 = delta_accum_1 + delta_2(2:end)' * a_1;
	delta_accum_2 = delta_accum_2 + delta_3' * a_2;
end;

Theta1_grad = delta_accum_1 / m;
Theta2_grad = delta_accum_2 / m;

 
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m).*Theta1(:,2:end);
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m).*Theta2(:,2:end);















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

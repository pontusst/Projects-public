function neurons = forwardPropagation(x, w, c)

[~, cols] = size(w);
bias = w(:,cols:cols);
w = w(:,1:cols-1);
neurons =(w*x') - bias;
neurons = activationFunction(c, neurons);
end
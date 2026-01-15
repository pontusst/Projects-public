function neurons = forwardPropegation(x, w)

[~, cols] = size(w);
bias = w(:,cols-1:cols);
neurons =(w*x) - bias;
neurons = activationFunction(neurons);
end
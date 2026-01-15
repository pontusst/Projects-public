function res = activationFunction(c , x)

res = zeros(1,length(x));

for i =1:length(x)
    res(i) = 1/(1+exp(-c*x(i)));
end
end
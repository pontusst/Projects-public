% nIn = the number of inputs
% nHidden = the number of hidden neurons
% nOut = the number of output neurons
% Weights (and biases) should take values in the range [-wMax,wMax]

function [wIH, wHO] = DecodeChromosome(chromosome, nIn, nHidden, nOut, wMax)

wIH = zeros(nHidden, nIn+1);
wHO = zeros(nOut, nHidden+1);
index = 1;

for i = 1: length(chromosome)
    chromosome(i) = chromosome(i) + wMax;
    chromosome(i) = chromosome(i)/(2*wMax);
end
chromosome = (chromosome).*(wMax - (-wMax));
chromosome = chromosome-wMax;

for i = 1:nHidden % iterate over rows in wIH
    wIH(i,:) = chromosome(1, index:index+nIn); % row 1 = chromosome index to index + row lenght.
    index = index + nIn+1;    
end
for i = 1:nOut % iterate over rows in wHO
    wHO(i,:) = chromosome(1, index:index+nHidden); % row 1 = chromosome index to index + row lenght.
    index = index + nHidden+1;    
end

%wI%H = min(max(wIH, -wMax), wMax);
%wHO = min(max(wHO, -wMax), wMax);
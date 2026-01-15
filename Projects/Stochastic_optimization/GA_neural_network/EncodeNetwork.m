function chromosome = EncodeNetwork(wIH, wHO, wMax)


[rowWih, colWih] = size(wIH);
[rowWho, colWho] = size(wHO);
chromosome = [];%;zeros(rowWih*colWih+rowWho*colWho);

for row = 1:rowWih
    chromosome = [chromosome wIH(row,:)]; % last element of row should be bias
end
for row = 1:rowWho
    chromosome = [chromosome wHO(row,:)];
end
%mean(chromosome)
%var(chromosome)

chromosome = (chromosome+wMax)./(wMax - (-wMax));
for i = 1:length(chromosome)
    chromosome(i) = -wMax + 2*wMax*chromosome(i);
end
%chromosome = (chromosome-mean(chromosome))./sqrt(var(chromosome));
%chromosome = (chromosome - (min(chromosome))) / (max(chromosome) - (min(chromosome)));
%chromosome = chromosome*(wMax - (-wMax)) + (-wMax);
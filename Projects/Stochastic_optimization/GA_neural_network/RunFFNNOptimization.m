clear all;

c = 1; % sigmoidConstant 
aMax = 20;
vMax = 25;
vMin = 1;

maximumFitness = 0;
maxMeanFitness = 0;
maxFitnessVal = 0;

tournamentSelectionParameter = 0.8; % high means less exploration
crossoverProbability = 0.7; % high means more exploration
mutationProbability = 0.020; % high means more exploration
nrOfGenerations = 2000; 

maxFitnessTrainPlot = zeros(1, nrOfGenerations);
maxFitnessValPlot = zeros(1, nrOfGenerations);

EPOCH = 5;
batchSize = 10;

deltaT = 0.1; % timestep
totalTime = 500; 
t = 0:deltaT:totalTime;

tbAmb = 283; % initial value for temperature of brakes
deltaTb = 0; % the change from ambient temparature
tbMax = 750;

% initialize weights 
minVal = -5;
maxVal = 5;
wMax = 5;

nIn = 3;
nHidden = 6;
nOut = 2;

sizeOfPopulation = 20;
sizeOfChromosome = nHidden*(nIn+1)+nOut*(nHidden+1);

population = zeros(sizeOfPopulation, sizeOfChromosome);

meanFitnessScores = zeros(sizeOfPopulation,1);
fitnessScores = zeros(batchSize,1);
fitnessVal = zeros(5,1);

for i = 1:sizeOfPopulation 
    wIH = minVal + (maxVal - minVal) * rand(nHidden, nIn+1); % every new slope gets new weights for simulation
    wHO = minVal + (maxVal - minVal) * rand(nOut, nHidden+1);
    population(i,:) = EncodeNetwork(wIH, wHO, wMax);
end

for iGeneration = 1:nrOfGenerations
    maximumFitness = 0;
    bestIndividualIndex = 0;
    maxMeanFitness = 0;
    for j = 1:sizeOfPopulation % for every chromosome/neuralnet we need to restart the simulation with new inital values
        [wIH, wHO] = DecodeChromosome(population(j,:), nIn,nHidden,nOut,wMax);

        %%% TRAINING SET %%
        for batch = 1:batchSize % training each chromosome on all slopes in training set
            iDataset = 1;
            slope = randi([1 10]);

            initialGear = 7;
            gearChange = 0;
            currentGear = initialGear+gearChange;
    
            vArr = zeros(size(t)); % velocity vector
            posArr = zeros(size(t)); % position vector
            tbArr = tbAmb*ones(size(t));  % the vector containing brake temperatures = tbAmb+deltaTb
            aArr = zeros(size(t));
    
            tStamp(1) = -20; % assumes the gear was changed 20 seconds ago. Just a dummy value not to interfere with gear change at start. 
            vArr(1) = 20;
            tbArr(1) = 500;

            i = 1; 
    
            while true % main simulation loop for every chromosome
                x = [vArr(i)/vMax, aArr(i)/aMax, tbArr(i)/tbMax];
                vH = forwardPropagation(x, wIH, c);
                vOut = forwardPropagation(vH, wHO, c); % in range 0 to 1
    
                if t(i) - tStamp < 2 % check if it was two second ago the last gear change was
                    vOut(2) = 0.5;
                end
                if t(i) >= 500
                    break;
                end
     
                previousGear = currentGear; % is needed to see if gear was changed
    
                [a , v, tb, pos, currentGear] = truckModel(vOut(1), vOut(2), deltaT, tbAmb, deltaTb, tbMax, vArr(i), posArr(i), tbArr(i), currentGear, slope, iDataset); % one step in simulation
                
                % update vectors
                vArr(i+1) = v; 
                posArr(i+1) = pos;
                tbArr(i+1) = tb; 
                aArr(i+1) = a;
    
                if previousGear ~= currentGear
                    tStamp = t(i);
                end
    
                i = i+1;
    
                if a > aMax || v > vMax || tb > tbMax || pos > 1000 || v < vMin
                    break
                end
            end
            fitnessScores(batch) = mean(vArr(1,1:i))*pos; % calculates the score for a chromosome for each of the slopes. 
        end
        meanFitnessScores(j) = mean(fitnessScores); % takes the mean of the above 10 scores
        if meanFitnessScores(j) > maxMeanFitness % saves the network if it is the best out of all of the chromosomes in that generation
            maxMeanFitness = meanFitnessScores(j);
            bestIndividualIndex = j;
            bestWIH = wIH;
            bestWHO = wHO;
            maxFitnessTrainPlot(iGeneration) = maxMeanFitness;

        end
     end

        %%% VALIDATION %%% 
        %%% Running validation test on the best network of current
        %%% generation %%%
        for slope = 1:5
            iDataset = 2;

            initialGear = 7;
            gearChange = 0;
            currentGear = initialGear+gearChange;

            i = 1; 

            vArrVal = zeros(size(t)); % velocity vector
            posArrVal = zeros(size(t)); % position vector
            tbArrVal = tbAmb*ones(size(t));  % the vector containing brake temperatures = tbAmb+deltaTb
            aArrVal = zeros(size(t));

            vArrVal(1) = 20;
            tbArrVal(1) = 500;

            while true
                xVal = [vArrVal(i)/vMax, aArrVal(i)/aMax, tbArrVal(i)/tbMax];
                vHVal = forwardPropagation(xVal, bestWIH, c);
                vOutVal = forwardPropagation(vHVal, bestWHO, c); % in range 0 to 1

                if t(i) - tStamp < 2 % check if it was two second ago the last gear change was
                    vOut(2) = 0.5;
                end

                previousGear = currentGear;

                [a , v, tb, pos, currentGear] = truckModel(vOutVal(1), vOutVal(2), deltaT, tbAmb, deltaTb, tbMax, vArrVal(i), posArrVal(i), tbArrVal(i), currentGear, slope, iDataset);

                vArrVal(i+1) = v;
                posArrVal(i+1) = pos;
                tbArrVal(i+1) = tb; 
                aArrVal(i+1) = a;

                if previousGear ~= currentGear
                    tStamp = t(i);
                end
    
                i = i+1;
    
                if a > aMax || v > vMax || tb > tbMax || pos > 1000 || v < vMin
                    break
                end
            end
            fitnessVal(slope) = mean(vArrVal(1,1:i))*pos; % calculate the score of each slope for best network. 
        end
        meanFitnessVal = mean(fitnessVal);
        maxFitnessValPlot(iGeneration) = meanFitnessVal;
        if meanFitnessVal > maxFitnessVal % save network if it preforms the best on validationset out of all the generations
            bestWIHVal = bestWIH;
            bestWHOVal = bestWHO;
            maxFitnessVal = meanFitnessVal;
            bestChromosome = EncodeNetwork(bestWIHVal, bestWHOVal, wMax);
        end
    
        %%% UPDATE POPULATION %%%
        tempPopulation = population;
    
        for i = 1:2:sizeOfPopulation
            i1 = TournamentSelect(fitnessScores,tournamentSelectionParameter);
            i2 = TournamentSelect(fitnessScores,tournamentSelectionParameter);
            chromosome1 = population(i1, :);
            chromosome2 = population(i2, :);
    
            r = rand;
            if (r < crossoverProbability)
                newChromosomePair = Cross(chromosome1, chromosome2);
                tempPopulation(i,:) = newChromosomePair(1,:);
                tempPopulation(i+1,:) = newChromosomePair(2,:);
            else
                tempPopulation(i,:) = chromosome1;
                tempPopulation(i+1,:) = chromosome2;
            end
        end
        for i = 1:sizeOfPopulation
            originalChromosome = tempPopulation(i,:);
            mutatedChromosome = Mutate(originalChromosome, mutationProbability);
            tempPopulation(i,:) = mutatedChromosome;
        end
    
        tempPopulation(1,:) = population(bestIndividualIndex,:);
        population = tempPopulation; 
end

%writematrix(bestChromosome, 'bestChromosome.csv') % save best chromosome/network to csv file

figure;

subplot(1,2,1); 
plot(1:iGeneration, maxFitnessTrainPlot);
xlabel('iGeneration');
ylabel('Maximum fitness of Training');
title('Maximum fitness of training as a function of generation');


subplot(1,2,2); 
plot(1:iGeneration, maxFitnessValPlot);
xlabel('iGeneration');
ylabel('Maximum fitness of Validation');
title('Maximum fitness of validation as a function of generation');

beep;

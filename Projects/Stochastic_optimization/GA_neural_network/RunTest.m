clear all;

minVal = -5;
maxVal = 5;
wMax = 5;

tbAmb = 283;
c = 1; % sigmoidConstant 
aMax = 20;
vMax = 25;
vMin = 1;
tbMax = 750;
deltaTb = 0;

nIn = 3;
nHidden = 6;
nOut = 2;

deltaT = 0.1;
totalTime = 200; 
t = 0:deltaT:totalTime;

% choose slope
iSlope = randi([1 10]);
iDataSet = 3;

% load chromosome
bestChromosome = readmatrix('bestChromosome.csv');

% decode chromosome 
[wIH, wHO] = DecodeChromosome(bestChromosome, nIn, nHidden, nOut, wMax);

initialGear = 7;
gearChange = 0;
currentGear = initialGear+gearChange;

vArr = zeros(size(t)); % velocity vector
posArr = zeros(size(t)); % position vector
tbArr = tbAmb*ones(size(t));  % the vector containing brake temperatures = tbAmb+deltaTb
angleArr = zeros(size(t));
ppArr = zeros(size(t));
gearArr = zeros(size(t));
aArr = zeros(size(t));

i = 1; 

tStamp = -20; %just to get it started
vArr(1) = 20;
tbArr(1) = 500;

while true % main simulation loop
    x = [vArr(i)/vMax, aArr(i)/aMax, tbArr(i)/tbMax];
    vH = forwardPropagation(x,wi, c);
    vOut = forwardPropagation(vH, wh, c); % in range 0 to 1

    if t(i) - tStamp < 2 % check if it was two second ago the last gear change was
        vOut(2) = 0.5;
    end

    previousGear = currentGear; % is needed to see if gear was changed
    alfa = GetSlopeAngle(posArr(i), iSlope, iDataSet);

    [a , v, tb, pos, currentGear] = truckModel(vOut(1), vOut(2), deltaT, tbAmb, deltaTb, tbMax, vArr(i), posArr(i), tbArr(i), currentGear, iSlope, iDataSet); % one step in simulation
    
    % update vectors
    vArr(i+1) = v; 
    posArr(i+1) = pos;
    tbArr(i+1) = tb; 
    aArr(i+1) = a;
    gearArr(i+1) = currentGear;
    angleArr(i+1) = alfa;
    ppArr(i+1) = vOut(1);

    if previousGear ~= currentGear
        tStamp = t(i);
    end

    i = i+1;

    if a > aMax || v > vMax || tb > tbMax || pos > 1000 || v < vMin
        break
    end
end

figure;

subplot(3,2,1); 
plot(posArr(1,1:i), angleArr(1,1:i));
xlabel('Position (m)');
ylabel('Angle of slope (degree)');
title('Slope angle as a function of position');


subplot(3,2,2); 
plot(posArr(1,1:i), vArr(1,1:i));
xlabel('Position (m)');
ylabel('Speed (m/s)');
title('Truck Speed as a function of position');


subplot(3,2,3); 
plot(posArr(1,1:i), ppArr(1,1:i));
xlabel('Position (m)');
ylabel('Brake pedal pressure');
title('Brake pedal pressure as a function of position');

subplot(3,2,4); 
plot(posArr(1,1:i), tbArr(1,1:i));
xlabel('Position (m)');
ylabel('Brake Temperature (K)');
title('Brake Temperature as a function of position');

subplot(3,2,5); 
plot(posArr(1,1:i), gearArr(1,1:i));
xlabel('Position (m)');
ylabel('Gear');
title('Gear choice as a function of position');


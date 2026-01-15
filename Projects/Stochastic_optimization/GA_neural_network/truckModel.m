function [aNext, vNext, tbNext, posNext, currentGear] = truckModel(pp, gearChange, deltaT, tbAmb, deltaTb, tbMax, v, pos, tb, currentGear, slope, iDataSet)

% define constants 
cb = 3000; % constant for gear brakes
ch = 40; % constant for brake temparature
tau = 30; % constant for cooling time
m = 20000;

g = 9.81; 

% engine brake force 
feb = [7.0*cb ; 5.0*cb ; 4.0*cb ; 3.0*cb ; 2.5*cb ; 2*cb ; 1.6*cb ; 1.4*cb ; 1.2*cb ; cb];

    % get angle of slope at current position
    alfa =  GetSlopeAngle(pos, slope, iDataSet);  % function value = angle of slope
    alfa = (alfa*2*pi)/360; % sin(x) takes radian needs to convert alfa(degrees) to alfa(radians)
    if alfa > 10
        alfa = 10;
    elseif alfa < 0
        alfa = 0;
    end

    % calculate force of gravity in direction of motion 
    fg = abs(m*g*sin(alfa));

    % calculate the foundational brake force 
    if tb < (tbMax-100)
        fb = (m*g*pp)/20; 
    else 
        eq1 = (tb-(tbMax-100))/100;
        eq2 = 1/exp(eq1);
        fb = ((m*g*pp)/20)*(eq2);
    end

    % determine engine brake force based on gear choice
    if gearChange <= 1/3
        gearChange = -1;
    elseif gearChange <= 2/3
        gearChange = 0;
    else
        gearChange = 1;
    end

    if currentGear+gearChange <= 10 && currentGear+gearChange > 0
        febUsed = feb(currentGear+gearChange);
        currentGear = currentGear+gearChange;
    else
        febUsed = feb(currentGear);
    end

    % calculate next step velocity
    vNext = v + deltaT*(fg - fb - febUsed)/m;

    % calculate next position
    posNext = pos + deltaT*v;

    % update brake temparature
    if pp < 0.01
        tbNext = tb - (tb - tbAmb) * deltaT / tau;  % cooling 
    else
        tbNext = tb + ch * pp * deltaT;  % heating 
    end

    % find acceleration 
    aNext = (fg - fb - febUsed)/m;

end
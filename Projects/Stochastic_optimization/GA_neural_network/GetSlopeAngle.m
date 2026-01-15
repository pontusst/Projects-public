%% This file provides the FORMAT you should use for the
%% slopes in HP2.3. x denotes the horizontal distance
%% travelled (by the truck) on a given slope, and
%% alpha measures the slope angle at distance x
%%
%% iSlope denotes the slope index (i.e. 1,2,..10 for the
%% training set etc.)
%% iDataSet determines whether the slope under consideration
%% belongs to the training set (iDataSet = 1), validation
%% set (iDataSet = 2) or the test set (iDataSet = 3).
%%
%% Note that the slopes given below are just EXAMPLES.
%% Please feel free to implement your own slopes below,
%% as long as they fulfil the criteria given in HP2.3.
%%
%% You may remove the comments above and below, as they
%% (or at least some of them) violate the coding standard 
%%  a bit. :)
%% The comments have been added as a clarification of the 
%% problem that should be solved!).


function alpha = GetSlopeAngle(x, iSlope, iDataSet)

if (iDataSet == 1)                                % Training
 if (iSlope == 1) 
   alpha = 4 + sin(x/100) + cos(sqrt(2)*x/50);    % You may modify this!

 elseif (iSlope == 2) 
   alpha = 4 + 2*sin(x/100) - cos(sqrt(2)*x/50);

 elseif (iSlope == 3) 
   alpha = 4 + 2*sin(x/10) + cos(sqrt(4)*(x/5));

 elseif (iSlope == 4) 
   alpha = 4 + 2*sin(x/50) + cos(sqrt(10)*x/50);

 elseif (iSlope == 5) 
   alpha = 5.5 + 2*cos(x/50) + cos(sqrt(8)*x/50);

 elseif (iSlope == 6)
   alpha = 4 + 2*sin(x/100) + cos(x/100); 

 elseif (iSlope == 7) 
   alpha = 4 + 2*sin((x/100)-15) + 2*cos((sqrt(2)*x/50)+50);

 elseif (iSlope == 8) 
   alpha = 4 + sin((x/10)+90) + 2*cos(x/20) - sin((x/40)-45) + cos(x/60);

 elseif (iSlope == 9) 
   alpha = 5 + 3*sin(x/10) + sin((x/5)-50) - cos(sqrt(2)*x/50);

 elseif (iSlope== 10)
    alpha = 5 + sin(x/80) + sin(x/170) - sin(x/220);
 end 


elseif (iDataSet == 2)                            % Validation
 if (iSlope == 1) 
   alpha = 6 - sin(x/100) + cos(sqrt(3)*x/50);    % You may modify this!

 elseif (iSlope == 2) 
   alpha = 5.5 - sin(x/160) + sin((x/100)) + cos(sqrt(3)*x/60);

 elseif (iSlope == 3) 
   alpha = 5 + sin((x/100)+45) + (cos(sqrt(3)*x/50));

 elseif (iSlope == 4) 
   alpha = 5.8 - cos(x/200) + sin(sqrt(3)*x/50) - cos(x/50);

 elseif (iSlope == 5) 
   alpha = 5 + sin(x/50) + cos(sqrt(5)*x/50);    % You may modify this!
 end 

elseif (iDataSet == 3)                           % Test
 if (iSlope == 1) 
   alpha = 6 - sin(x/100) - cos(sqrt(7)*x/50);   % You may modify this!

 elseif (iSlope == 2) 
   alpha = 5 + (x/1000) + cos(x/180) + cos(sqrt(5)*(x/150));

 elseif (iSlope == 3) 
   alpha = 4 + sin((x/30)+45) + 2*cos(x/45) - sin((x/40)-45) + cos(x/160);

 elseif (iSlope == 4) 
   alpha = 5 + sin(x/150) - cos(x/270) + sin(sqrt(5)*x/90);

 elseif (iSlope == 5)
   alpha = 4 + (x/1000) + sin(x/70) + cos(sqrt(7)*x/100); % You may modify this!

 end

end

end

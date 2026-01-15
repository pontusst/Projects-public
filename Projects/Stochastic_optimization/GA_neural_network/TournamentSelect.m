function iSelected = TournamentSelect(fitnessScores, pTournament)

populationSize = size(fitnessScores, 1);

iTmp1 = 1 + fix(rand*populationSize);
iTmp2 = 1 + fix(rand*populationSize);

r = rand;

if (r < pTournament)
    if (fitnessScores(iTmp1) > fitnessScores(iTmp2))
        iSelected = iTmp1;
    else 
        iSelected = iTmp2;
    end
else 
    if (fitnessScores(iTmp1) > fitnessScores(iTmp2))
        iSelected = iTmp2;
    else
        iSelected = iTmp1;
    end
end
end
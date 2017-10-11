% Bayesian (posterior predictive estimate) for thetajc.
function thetajc = Thetajc(a)
%loading the mat file
    load('spamData.mat');
%binarize the Xtrain
    BXtrain=binarization(Xtrain);
    counterc0 = 0;
    counterc1 = 0;
    counterj1 = 0;
    counterj0 = 0;
    thetajc = zeros(2,57);
    %a = 0; 
    
%theta(j, c=1)
    for column = 1:57
        counterc1 = 0;
        counterj1 = 0;
        for row = 1:3065
            if ytrain(row, 1) == 1
                counterc1 = counterc1 + 1;
                if BXtrain(row, column) == 1
                   counterj1 = counterj1 + 1;
                end
            end
        end
        thetajc(1,column) = (counterj1 + a)./(counterc1 + 2*a);
    end
    
%theta(j, c=0)
    for column = 1:57
        counterc0 = 0;
        counterj0 = 0;
        for row = 1:3065
            if ytrain(row, 1) == 0
                counterc0 = counterc0 + 1;
                if BXtrain(row, column) == 1
                    counterj0 = counterj0 + 1;
                end
            end
        end
        thetajc(2,column) = (counterj0 + a)./(counterc0 + 2*a);
    end    
end
    
    

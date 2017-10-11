%% =========Classifer4 K-Nearest Neighbors=========%%
clc;
close all;
clear all;

%loading the mat file
load('spamData.mat');

%% =====================preprocessing features====================%
     %======binarization=======%
%==for Xtrain==%
for row = 1:3065
    for column = 1:57
        if Xtrain(row, column) == 0
            BXtrain(row, column) = 0;
        else
            BXtrain(row, column) = 1;
        end
    end   
end
%==for Xtest==%
for row = 1:1536
    for column = 1:57
        if Xtest(row, column) == 0
            BXtest(row, column) = 0;
        else
            BXtest(row, column) = 1;
        end
    end   
end

%% ============ KNN -- Binarization -- Hamming distance =========%%

      %%===========For test error=================%%

%calculate the Hamming distance
distance = zeros(1536, 3065);
for rowtest = 1:1536
    for rowtrain = 1:3065
        d(1, :) = BXtest(rowtest, :) - BXtrain(rowtrain, :);
        dis = length( find(d(1, :)));
        distance(rowtest, rowtrain) = dis;
    end
end    

    %find the K shortest distances    
error_column = 0;
for K = [(1:1:10), (15:5:100)]
    error_column = error_column + 1;
    for row = 1:1536
        [Ndis, index] = sort(distance(row, :));
        for i = 1:K
            Ytest(row, i) = ytrain (index(1, i), 1);
        end
    end
    
    %calculate the rate of 1 in K and 0 in K
    for row = 1:1536
        counter = 0;
        for i = 1:K
            if Ytest(row, i) == 1
                counter = counter + 1;
            end
        end
        Btest_p1(row, 1) = counter / K;
        Btest_p0(row, 1) = 1 - counter / K;
    end
    
    %classify the final class for BXtest
    for row = 1:1536
        if Btest_p1(row, 1) > Btest_p0(row, 1)
            BYtest(row, 1) = 1;
        else
            BYtest(row, 1) = 0;
        end
    end
    
    %calculate the error rate
    error = 0;
    for row = 1:1536
        if BYtest(row, 1) ~= ytest(row, 1)
            error = error + 1;
        end
    end
    errorrate_BXtest(1, error_column) = error / 1536; 
    if K == 1 
        disp('errorrate_BXtest_K=1: ')
        disp(errorrate_BXtest(1, error_column))
    end    
    if K == 10 
        disp('errorrate_BXtest_K=10: ')
        disp(errorrate_BXtest(1, error_column))
    end     
    if K == 100 
        disp('errorrate_BXtest_K=100: ')
        disp(errorrate_BXtest(1, error_column))
    end
end
figure(1);
scatter([(1:1:10), (15:5:100)], errorrate_BXtest, 'red');
hold on;
figure(2);
plot([(1:1:10), (15:5:100)], errorrate_BXtest, 'red');
hold on;

 %%===========For train error=================%%

%calculate the Hamming distance
distance = zeros(3065, 3065);
for rowtest = 1:3065
    for rowtrain = 1:3065
        d(1, :) = BXtrain(rowtest, :) - BXtrain(rowtrain, :);
        dis = length( find(d(1, :)));
        distance(rowtest, rowtrain) = dis;
    end
end    

    %find the K shortest distances    
error_column = 0;
for K = [(1:1:10), (15:5:100)]
    error_column = error_column + 1;
    for row = 1:3065
        [Ndis, index] = sort(distance(row, :));
        for i = 1:K
            Ytrain(row, i) = ytrain (index(1, i), 1);
        end
    end
    
    %calculate the rate of 1 in K and 0 in K
    for row = 1:3065
        counter = 0;
        for i = 1:K
            if Ytrain(row, i) == 1
                counter = counter + 1;
            end
        end
        Btrain_p1(row, 1) = counter / K;
        Btrain_p0(row, 1) = 1 - counter / K;
    end
    
    %classify the final class for BXtest
    for row = 1:3065
        if Btrain_p1(row, 1) > Btrain_p0(row, 1)
            BYrain(row, 1) = 1;
        else
            BYrain(row, 1) = 0;
        end
    end
    
    %calculate the error rate
    error = 0;
    for row = 1:3065
        if BYrain(row, 1) ~= ytrain(row, 1)
            error = error + 1;
        end
    end
    errorrate_BXtrain(1, error_column) = error / 3065; 
    if K == 1 
        disp('errorrate_BXtrain_K=1: ')
        disp(errorrate_BXtrain(1, error_column))
    end    
    if K == 10 
        disp('errorrate_BXtrain_K=10: ')
        disp(errorrate_BXtrain(1, error_column))
    end     
    if K == 100 
        disp('errorrate_BXtrain_K=100: ')
        disp(errorrate_BXtrain(1, error_column))
    end
end
figure(1);
scatter([(1:1:10), (15:5:100)], errorrate_BXtrain, 'black');
legend('KNN-BXtest-errorrate', 'KNN-BXtrain-errorrate'); 
title('KNN-Binarization-errorrate');
figure(2);
plot([(1:1:10), (15:5:100)], errorrate_BXtrain, 'black');
legend('KNN-BXtest-errorrate', 'KNN-BXtrain-errorrate');  
title('KNN-Binarization-errorrate');



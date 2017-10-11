%% =========Classifer4 K-Nearest Neighbors=========%%
clc;
close all;
clear all;
%loading the mat file
load('spamData.mat');

%% =====================preprocessing features====================%
     %======Log-transform======%
%==for Xtrain==%
for row = 1:3065
    for column = 1:57
        LXtrain(row, column) = log(Xtrain(row, column) + 0.1);
    end
end
%==for Xtest==%
for row = 1:1536
    for column = 1:57
        LXtest(row, column) = log(Xtest(row, column) + 0.1);
    end
end

%% ============ KNN -- Binarization -- Hamming distance =========%%
      %%===========For test error=================%%
%calculate the Euclidean distance
distance = zeros(1536, 3065);
for rowtest = 1:1536
    for rowtrain = 1:3065
        distance(rowtest, rowtrain) = sqrt(sum((LXtest(rowtest, :) - LXtrain(rowtrain, :)).^2));
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
    
    %calculate the rate that LXtest classified to class 1 and class 0
    for row = 1:1536
        counter = 0;
        for i = 1:K
            if Ytest(row, i) == 1
                counter = counter + 1;
            end
        end
        Ltest_p1(row, 1) = counter / K;
        Ltest_p0(row, 1) = 1 - counter / K;
    end
    
    %classify the final class for LXtest
    for row = 1:1536
        if Ltest_p1(row, 1) > Ltest_p0(row, 1)
            LYtest(row, 1) = 1;
        else
            LYtest(row, 1) = 0;
        end
    end
    
    %calculate the error rate
    error = 0;
    for row = 1:1536
        if LYtest(row, 1) ~= ytest(row, 1)
            error = error + 1;
        end
    end
    errorrate_LXtest(1, error_column) = error / 1536; 
    if K == 1 
        disp('errorrate_LXtest_K=1: ')
        disp(errorrate_LXtest(1, error_column))
    end    
    if K == 10 
        disp('errorrate_LXtest_K=10: ')
        disp(errorrate_LXtest(1, error_column))
    end     
    if K == 100 
        disp('errorrate_LXtest_K=100: ')
        disp(errorrate_LXtest(1, error_column))
    end
end
figure(1);
scatter([(1:1:10), (15:5:100)], errorrate_LXtest, 'red');
hold on;
figure(2);
plot([(1:1:10), (15:5:100)], errorrate_LXtest, 'red');
hold on;

         %%==================For train error=====================%%

%calculate the Euclidean distance
distance = zeros(3065, 3065);
for rowtest = 1:3065
    for rowtrain = 1:3065
        distance(rowtest, rowtrain) = sqrt(sum((LXtrain(rowtest, :) - LXtrain(rowtrain, :)).^2));
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
    
    %calculate the rate that LXtrain classified to class 1 and class 0
    for row = 1:3065
        counter = 0;
        for i = 1:K
            if Ytrain(row, i) == 1
                counter = counter + 1;
            end
        end
        Ltrain_p1(row, 1) = counter / K;
        Ltrain_p0(row, 1) = 1 - counter / K;
    end
    
    %classify the final class for LXtrain
    for row = 1:3065
        if Ltrain_p1(row, 1) > Ltrain_p0(row, 1)
            LYrain(row, 1) = 1;
        else
            LYrain(row, 1) = 0;
        end
    end
    
    %calculate the error rate
    error = 0;
    for row = 1:3065
        if LYrain(row, 1) ~= ytrain(row, 1)
            error = error + 1;
        end
    end
    errorrate_LXtrain(1, error_column) = error / 3065; 
    if K == 1 
        disp('errorrate_LXtrain_K=1: ')
        disp(errorrate_LXtrain(1, error_column))
    end    
    if K == 10 
        disp('errorrate_LXtrain_K=10: ')
        disp(errorrate_LXtrain(1, error_column))
    end     
    if K == 100 
        disp('errorrate_LXtrain_K=100: ')
        disp(errorrate_LXtrain(1, error_column))
    end
end
figure(1);
scatter([(1:1:10), (15:5:100)], errorrate_LXtrain, 'black');
legend('KNN-BXtest-errorrate', 'KNN-LXtrain-errorrate'); 
title('KNN-Log transform-errorrate');
figure(2);
plot([(1:1:10), (15:5:100)], errorrate_LXtrain, 'black');
legend('KNN-BXtest-errorrate', 'KNN-LXtrain-errorrate');  
title('KNN-Log transform-errorrate');
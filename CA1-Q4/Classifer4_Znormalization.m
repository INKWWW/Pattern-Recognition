%% =========Classifer4 K-Nearest Neighbors=========%%
clc;
close all;
clear all;
%loading the mat file
load('spamData.mat');

%% =====================preprocessing features====================%
     %====Z-normalizatio====%
%====Z-normalization for Xtrain====%
ZXtrain = zscore(Xtrain);
for row = 1:3065
    for column = 1:57
        ZXtrain(row, column) = ZXtrain(row, column);
    end
end

%====Z-normalization for Xtest====%
meantrain = mean(Xtrain);
stddevtrain = std(Xtrain);
for row = 1:1536
    for column = 1:57
        ZXtest(row, column) = (Xtest(row, column) - meantrain(1, column))/stddevtrain(1, column);
    end
end
%ZXtest = zscore(Xtest);

%% ============ KNN -- Binarization -- Hamming distance =========%%

      %%===================For test error===================%%
      
%calculate the Euclidean distance
distance = zeros(1536, 3065);
for rowtest = 1:1536
    for rowtrain = 1:3065
        distance(rowtest, rowtrain) = sqrt(sum((ZXtest(rowtest, :) - ZXtrain(rowtrain, :)).^2));
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
    
    %calculate the rate that ZXtest classified to class 1 and class 0
    for row = 1:1536
        counter = 0;
        for i = 1:K
            if Ytest(row, i) == 1
                counter = counter + 1;
            end
        end
        Ztest_p1(row, 1) = counter / K;
        Ztest_p0(row, 1) = 1 - counter / K;
    end
    
    %classify the final class for ZXtest
    for row = 1:1536
        if Ztest_p1(row, 1) > Ztest_p0(row, 1)
            ZYtest(row, 1) = 1;
        else
            ZYtest(row, 1) = 0;
        end
    end
    
    %calculate the error rate
    error = 0;
    for row = 1:1536
        if ZYtest(row, 1) ~= ytest(row, 1)
            error = error + 1;
        end
    end
    errorrate_ZXtest(1, error_column) = error / 1536; 
    if K == 1 
        disp('errorrate_ZXtest_K=1: ')
        disp(errorrate_ZXtest(1, error_column))
    end    
    if K == 10 
        disp('errorrate_ZXtest_K=10: ')
        disp(errorrate_ZXtest(1, error_column))
    end     
    if K == 100 
        disp('errorrate_ZXtest_K=100: ')
        disp(errorrate_ZXtest(1, error_column))
    end
end
figure(1);
scatter([(1:1:10), (15:5:100)], errorrate_ZXtest, 'red');
hold on;
figure(2);
plot([(1:1:10), (15:5:100)], errorrate_ZXtest, 'red');
hold on;

       %%====================For train error=====================%%

%calculate the Euclidean distance
distance = zeros(3065, 3065);
for rowtest = 1:3065
    for rowtrain = 1:3065
        distance(rowtest, rowtrain) = sqrt(sum((ZXtrain(rowtest, :) - ZXtrain(rowtrain, :)).^2));
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
    
    %calculate the rate that ZXtrain classified to class 1 and class 0
    for row = 1:3065
        counter = 0;
        for i = 1:K
            if Ytrain(row, i) == 1
                counter = counter + 1;
            end
        end
        Ztrain_p1(row, 1) = counter / K;
        Ztrain_p0(row, 1) = 1 - counter / K;
    end
    
    %classify the final class for ZXtrain
    for row = 1:3065
        if Ztrain_p1(row, 1) > Ztrain_p0(row, 1)
            ZYrain(row, 1) = 1;
        else
            ZYrain(row, 1) = 0;
        end
    end
    
    %calculate the error rate
    error = 0;
    for row = 1:3065
        if ZYrain(row, 1) ~= ytrain(row, 1)
            error = error + 1;
        end
    end
    errorrate_ZXtrain(1, error_column) = error / 3065; 
    if K == 1 
        disp('errorrate_ZXtrain_K=1: ')
        disp(errorrate_ZXtrain(1, error_column))
    end    
    if K == 10 
        disp('errorrate_ZXtrain_K=10: ')
        disp(errorrate_ZXtrain(1, error_column))
    end     
    if K == 100 
        disp('errorrate_ZXtrain_K=100: ')
        disp(errorrate_ZXtrain(1, error_column))
    end
end
figure(1);
scatter([(1:1:10), (15:5:100)], errorrate_ZXtrain, 'black');
legend('KNN-BXtest-errorrate', 'KNN-ZXtrain-errorrate');
title('KNN-Znormalization-errorrate');
figure(2);
plot([(1:1:10), (15:5:100)], errorrate_ZXtrain, 'black');
legend('KNN-BXtest-errorrate', 'KNN-ZXtrain-errorrate');
title('KNN-Znormalization-errorrate');
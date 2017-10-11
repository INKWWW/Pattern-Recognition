%% =========CA1-Q3 Logistic Regression -- Z-normalization=========%%
clc;
close all;
clear all;

%loading the mat file
load('spamData.mat');

%% =====================preprocessing features====================%
%==== preprocessing and add a column of 1 into features matrix
train = ones(3065, 58);
test = ones(1536, 58);

     %====Z-normalizatio====%
%====Z-normalization for Xtrain====%
ZXtrain = zscore(Xtrain);
for row = 1:3065
    for column = 1:57
        train(row, column+1) = ZXtrain(row, column);
    end
end
ZXtrain = train;
%====Z-normalization for Xtest====%
meantrain = mean(Xtrain);
stddevtrain = std(Xtrain);
for row = 1:1536
    for column = 1:57
        test(row, column+1) = (Xtest(row, column) - meantrain(1, column))/stddevtrain(1, column);
    end
end
ZXtest = test;
%ZXtest = zscore(Xtest);

%% =============calculate the error rate on Z-normalization ================%
%%===== test error rate on Z-normalization =====%%
Ytest_column = 0;
for lmd = [(1:1:10), (15:5:100)]
    Ytest_column = Ytest_column + 1;
    w = zeros(58, 1);
    I = eye(58); 
    margin = 1;
    lmdw = zeros(58, 1);
    
    %=====Using Newton's Method until convergence
    while (margin > 10^-15)        
        for row = 1:3065
            mu(row, 1) = 1/(1+exp(-(w'* ZXtrain(row, :)')));
            mu_y(row, 1) = mu(row, 1) - ytrain(row, 1);
            S(row, row) = mu(row, 1) * (1 - mu(row, 1));
        end
        for counter = 2:58
            lmdw(counter, 1) = lmd * w(counter, 1);
        end
        lmdw(1,1) = w(1,1);
        greg = ZXtrain' * mu_y + lmdw;
        Hreg = ZXtrain' * S * ZXtrain + lmd * I;
        w = w - (Hreg^-1) * greg;
        margin = ((Hreg^-1) * greg)' * ((Hreg^-1) * greg);
    end    
    
    %======calculate the error rate and plot for ZXtest  
  % Classify the LXtest to class 1 or class 0   
    for row = 1:1536
        p(row, 1) = w' * ZXtest(row, :)';
        if p(row, 1) > 0
            ZYtest(row, 1) = 1;
        else
            ZYtest(row, 1) = 0;
        end         
    end  
    errorcounter = 0;
    for row = 1:1536
        if ZYtest(row, 1) ~= ytest(row, 1)
           errorcounter = errorcounter + 1;
        end
    end       
    error_rate_Ztest(1, Ytest_column) = errorcounter / 1536;
        
    if lmd == 1 
        disp('error_rate_ZXtest_lmd=1: ')
        disp(error_rate_Ztest(1, Ytest_column))
    end    
    if lmd == 10 
        disp('error_rate_ZXtest_lmd=10: ')
        disp(error_rate_Ztest(1, Ytest_column))
    end     
    if lmd == 100 
        disp('error_rate_ZXtest_lmd=100: ')
        disp(error_rate_Ztest(1, Ytest_column))
    end 
end
figure(1);
scatter([(1:1:10), (15:5:100)], error_rate_Ztest, 'red');
hold on;
figure(2);
plot([(1:1:10), (15:5:100)], error_rate_Ztest, 'red');
hold on;


%% =============calculate the error rate on Z-normalization ================%
%%===== training error rate on Z-normalization =====%%
Ytrain_column = 0;
for lmd = [(1:1:10), (15:5:100)]
    Ytrain_column = Ytrain_column + 1;
    w = zeros(58, 1);
    I = eye(58); 
    margin = 1;
    lmdw = zeros(58, 1);
    
    %=====Using Newton's Method until convergence
    while (margin > 10^-15)        
        for row = 1:3065
            mu(row, 1) = 1/(1+exp(-(w'* ZXtrain(row, :)')));
            mu_y(row, 1) = mu(row, 1) - ytrain(row, 1);
            S(row, row) = mu(row, 1) * (1 - mu(row, 1));
        end
        for counter = 2:58
            lmdw(counter, 1) = lmd * w(counter, 1);
        end
        lmdw(1,1) = w(1,1);
        greg = ZXtrain' * mu_y + lmdw;
        Hreg = ZXtrain' * S * ZXtrain + lmd * I;
        w = w - (Hreg^-1) * greg;
        margin = ((Hreg^-1) * greg)' * ((Hreg^-1) * greg);
    end    
    
    %======calculate the error rate and plot for ZXtrain  
  % Classify the LXtest to class 1 or class 0    
    for row = 1:3065
        p(row, 1) = w' * ZXtrain(row, :)';
        if p(row, 1) > 0
            ZYtrain(row, 1) = 1;
        else
            ZYtrain(row, 1) = 0;
        end         
    end  
    errorcounter = 0;
    for row = 1:3065
        if ZYtrain(row, 1) ~= ytrain(row, 1)
           errorcounter = errorcounter + 1;
        end
    end       
    error_rate_Ztrain(1, Ytrain_column) = errorcounter / 3065;
        
    if lmd == 1
        disp('error_rate_ZXtrain_lmd=10: ')
        disp(error_rate_Ztrain(1, Ytrain_column))
    end    
    if lmd == 10 
        disp('error_rate_ZXtrain_lmd=10: ')
        disp(error_rate_Ztrain(1, Ytrain_column))
    end     
    if lmd == 100 
        disp('error_rate_ZXtrain_lmd=100: ')
        disp(error_rate_Ztrain(1, Ytrain_column))
    end 
end
figure(1);
scatter([(1:1:10), (15:5:100)], error_rate_Ztrain,'black');
title('error-rate-Z-normalization');
legend('Lg-error-rate-ZXtest', 'Lg-error-rate-ZXtrain');
figure(2);
plot([(1:1:10), (15:5:100)], error_rate_Ztrain,'black');
title('error-rate-Z-normalization');
legend('Lg-error-rate-ZXtest', 'Lg-error-rate-ZXtrain');
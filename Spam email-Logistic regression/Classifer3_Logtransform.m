%% =========CA1-Q3 Logistic Regression -- Log-transform =========%%
clc;
close all;
clear all;

%loading the mat file
load('spamData.mat');

%% =====================preprocessing features====================%
%==== preprocessing and add a column of 1 into features matrix
train = ones(3065, 58);
test = ones(1536, 58);

   %      %======Log-transform======%
%==for Xtrain==%
for row = 1:3065
    for column = 1:57
        train(row, column+1) = log(Xtrain(row, column) + 0.1);
    end
end
LXtrain = train;
%==for Xtest==%
for row = 1:1536
    for column = 1:57
        test(row, column+1) = log(Xtest(row, column) + 0.1);
    end
end
LXtest = test;

%% =============calculate the error rate on Log-transform ================%
%%===== test error rate on Log-transform =====%%
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
            mu(row, 1) = 1/(1+exp(-(w'* LXtrain(row, :)')));
            mu_y(row, 1) = mu(row, 1) - ytrain(row, 1);
            S(row, row) = mu(row, 1) * (1 - mu(row, 1));
        end
        for counter = 2:58
            lmdw(counter, 1) = lmd * w(counter, 1);
        end
        lmdw(1,1) = w(1,1);
        greg = LXtrain' * mu_y + lmdw;
        Hreg = LXtrain' * S * LXtrain + lmd * I;
        w = w - (Hreg^-1) * greg;
        margin = ((Hreg^-1) * greg)' * ((Hreg^-1) * greg);
    end    
    
    %======calculate the error rate and plot for LXtest  
  % Classify the LXtest to class 1 or class 0  
    for row = 1:1536
        p(row, 1) = w' * LXtest(row, :)';
        if p(row, 1) > 0
            LYtest(row, 1) = 1;
        else
            LYtest(row, 1) = 0;
        end         
    end  
    errorcounter = 0;
    for row = 1:1536
        if LYtest(row, 1) ~= ytest(row, 1)
           errorcounter = errorcounter + 1;
        end
    end       
    error_rate_Ltest(1, Ytest_column) = errorcounter / 1536;
        
    if lmd == 1 
        disp('error_rate_LXtest_lmd=1: ')
        disp(error_rate_Ltest(1, Ytest_column))
    end    
    if lmd == 10 
        disp('error_rate_LXtest_lmd=10: ')
        disp(error_rate_Ltest(1, Ytest_column))
    end     
    if lmd == 100 
        disp('error_rate_LXtest_lmd=100: ')
        disp(error_rate_Ltest(1, Ytest_column))
    end 
end
figure(1);
scatter([(1:1:10), (15:5:100)], error_rate_Ltest, 'red');
hold on;
figure(2);
plot([(1:1:10), (15:5:100)], error_rate_Ltest, 'red');
hold on;

%% =============calculate the error rate on Log-transform ================%
%%===== training error rate on Log-transform =====%%
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
            mu(row, 1) = 1/(1+exp(-(w'* LXtrain(row, :)')));
            mu_y(row, 1) = mu(row, 1) - ytrain(row, 1);
            S(row, row) = mu(row, 1) * (1 - mu(row, 1));
        end
        for counter = 2:58
            lmdw(counter, 1) = lmd * w(counter, 1);
        end
        lmdw(1,1) = w(1,1);
        greg = LXtrain' * mu_y + lmdw;
        Hreg = LXtrain' * S * LXtrain + lmd * I;
        w = w - (Hreg^-1) * greg;
        margin = ((Hreg^-1) * greg)' * ((Hreg^-1) * greg);
    end    
    
    %======calculate the error rate and plot for ZXtrain  
  % Classify the LXtest to class 1 or class 0   
    for row = 1:3065
        p(row, 1) = w' * LXtrain(row, :)';
        if p(row, 1) > 0
            LYtrain(row, 1) = 1;
        else
            LYtrain(row, 1) = 0;
        end         
    end  
    errorcounter = 0;
    for row = 1:3065
        if LYtrain(row, 1) ~= ytrain(row, 1)
           errorcounter = errorcounter + 1;
        end
    end       
    error_rate_Ltrain(1, Ytrain_column) = errorcounter / 3065;
        
    if lmd == 1
        disp('error_rate_LXtrain_lmd=1: ')
        disp(error_rate_Ltrain(1, Ytrain_column))
    end    
    if lmd == 10 
        disp('error_rate_LXtrain_lmd=10: ')
        disp(error_rate_Ltrain(1, Ytrain_column))
    end     
    if lmd == 100 
        disp('error_rate_LXtrain_lmd=100: ')
        disp(error_rate_Ltrain(1, Ytrain_column))
    end 
end
figure(1);
scatter([(1:1:10), (15:5:100)], error_rate_Ltrain,'black');
title('error-rate-Logtransform');
legend('Lg-error-rate-LXtest', 'Lg-error-rate-LXtrain');
figure(2);
plot([(1:1:10), (15:5:100)], error_rate_Ltrain,'black');
title('error-rate-Logtransform');
legend('Lg-error-rate-LXtest', 'Lg-error-rate-LXtrain');
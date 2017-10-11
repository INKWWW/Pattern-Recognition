%% =========CA1-Q3 Logistic Regression -- Bnarization=========%%
clc;
close all;
clear all;

%loading the mat file
load('spamData.mat');

%% =====================preprocessing features====================%
%==== preprocessing and add a column of 1 into features matrix
train = ones(3065, 58);
test = ones(1536, 58);

    %======binarization=======%
%==for Xtrain==%
for row = 1:3065
    for column = 1:57
        if Xtrain(row, column) == 0
            train(row, column+1) = 0;
        else
            train(row, column+1) = 1;
        end
    end   
end
BXtrain = train;
%==for Xtest==%
for row = 1:1536
    for column = 1:57
        if Xtest(row, column) == 0
            test(row, column+1) = 0;
        else
            test(row, column+1) = 1;
        end
    end   
end
BXtest = test;

%% =============calculate the error rate on Bnarization ================%
%%===== test error rate on Bnarization =====%%
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
            mu(row, 1) = 1/(1+exp(-(w'* BXtrain(row, :)')));
            mu_y(row, 1) = mu(row, 1) - ytrain(row, 1);
            S(row, row) = mu(row, 1) * (1 - mu(row, 1));
        end
        for counter = 2:58
            lmdw(counter, 1) = lmd * w(counter, 1);
        end
        lmdw(1,1) = w(1,1);
        greg = BXtrain' * mu_y + lmdw;
        Hreg = BXtrain' * S * BXtrain + lmd * I;
        w = w - (Hreg^-1) * greg;
        margin = ((Hreg^-1) * greg)' * ((Hreg^-1) * greg);
    end    
    
    %======calculate the error rate and plot for BXtest  
  % Classify the BXtest to class 1 or class 0  
    for row = 1:1536
        p(row, 1) = w' * BXtest(row, :)';
        if p(row, 1) > 0
            BYtest(row, 1) = 1;
        else
            BYtest(row, 1) = 0;
        end         
    end  
    errorcounter = 0;
    for row = 1:1536
        if BYtest(row, 1) ~= ytest(row, 1)
           errorcounter = errorcounter + 1;
        end
    end       
    error_rate_Btest(1, Ytest_column) = errorcounter / 1536;
        
    if lmd == 1 
        disp('error_rate_BXtest_lmd=1: ')
        disp(error_rate_Btest(1, Ytest_column))
    end    
    if lmd == 10 
        disp('error_rate_BXtest_lmd=10: ')
        disp(error_rate_Btest(1, Ytest_column))
    end     
    if lmd == 100 
        disp('error_rate_BXtest_lmd=100: ')
        disp(error_rate_Btest(1, Ytest_column))
    end 
end
figure(1);
scatter([(1:1:10), (15:5:100)], error_rate_Btest, 'red');
hold on;
figure(2);
plot([(1:1:10), (15:5:100)], error_rate_Btest, 'red');
hold on;

%% =============calculate the error rate on Bnarization ================%
%%===== training error rate on Bnarization =====%%
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
            mu(row, 1) = 1/(1+exp(-(w'* BXtrain(row, :)')));
            mu_y(row, 1) = mu(row, 1) - ytrain(row, 1);
            S(row, row) = mu(row, 1) * (1 - mu(row, 1));
        end
        for counter = 2:58
            lmdw(counter, 1) = lmd * w(counter, 1);
        end
        lmdw(1,1) = w(1,1);
        greg = BXtrain' * mu_y + lmdw;
        Hreg = BXtrain' * S * BXtrain + lmd * I;
        w = w - (Hreg^-1) * greg;
        margin = ((Hreg^-1) * greg)' * ((Hreg^-1) * greg);
    end    
    
    %======calculate the error rate and plot for BXtrain  
  % Classify the BXtest to class 1 or class 0  
    for row = 1:3065
        p(row, 1) = w' * BXtrain(row, :)';
        if p(row, 1) > 0
            BYtrain(row, 1) = 1;
        else
            BYtrain(row, 1) = 0;
        end         
    end  
    errorcounter = 0;
    for row = 1:3065
        if BYtrain(row, 1) ~= ytrain(row, 1)
           errorcounter = errorcounter + 1;
        end
    end       
    error_rate_Btrain(1, Ytrain_column) = errorcounter / 3065;
        
    if lmd == 1
        disp('error_rate_BXtrain_lmd=1: ')
        disp(error_rate_Btrain(1, Ytrain_column))
    end    
    if lmd == 10 
        disp('error_rate_BXtrain_lmd=10: ')
        disp(error_rate_Btrain(1, Ytrain_column))
    end     
    if lmd == 100 
        disp('error_rate_BXtrain_lmd=100: ')
        disp(error_rate_Btrain(1, Ytrain_column))
    end 
end
figure(1);
scatter([(1:1:10), (15:5:100)], error_rate_Btrain,'black');
title('error-rate-Binarization');
legend('Lg-error-rate-BXtest', 'Lg-error-rate-BXtrain');
figure(2);
plot([(1:1:10), (15:5:100)], error_rate_Btrain,'black');
title('error-rate-Binarization');
legend('Lg-error-rate-BXtest', 'Lg-error-rate-BXtrain');
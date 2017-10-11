%% ======Q1 Beta-bernoulli Naive Bayes ======%
%use the plug-in ML estimate for p(y) and 
%the Bayesian (posterior predictive estimate) for p(x|y).
clc;
close all;
clear all;

%loading the mat file
load('spamData.mat');

%% ========binarize the Xtest=======%
BXtest = binarization(Xtest);
BXtrain = binarization(Xtrain);
a = [0:0.5:100];
errortest = zeros(1, 201);
errortrain = zeros(1, 201);
Ytestedcolumn = 0;

%% ===========calculate paic1=p(y=c|T)===========%
Nc1 = 0;
N = 3065;
%paic1
for row = 1:3065
    if ytrain(row,1) == 1
        Nc1 = Nc1 + 1;        
    end
end
paic1 = Nc1 / N;

%% =====calculate the Bayesian posterior predictive estimate=====%

%=========For testing=======%
Sx = size(BXtest);
for a = 0:0.5:100
     thetajc = Thetajc(a);
     Ytestedcolumn = Ytestedcolumn + 1;
     logp = zeros(Sx(1,1) ,2);
     Yresult = zeros(Sx(1,1) ,1);
     %====probability of judging them to class 1====%
     for row = 1:Sx(1,1)
         mc1 = 0;
         for column = 1:Sx(1,2)
             xj = BXtest(row, column);
             if xj == 1
                mc1 = mc1 + log(thetajc(1, column) );
             else
                mc1 = mc1 + log((1 - thetajc(1,column)) );
             end
             logp(row, 1) = log(paic1) + mc1;             
         end
     end         
     %====probability of judging them to class 0====%
     for row = 1:Sx(1,1)
         mc0 = 0;
         for column = 1:Sx(1,2)
             xj = BXtest(row, column);
             if xj == 1
                mc0 = mc0 + log(thetajc(2, column) );
             else
                mc0 = mc0 + log((1 - thetajc(2, column)) );
             end
             logp(row, 2) = log(1-paic1) + mc0;             
         end 
     end     
     %===determine the class 0/1 by comparing====%
     for row = 1:Sx(1,1)
         if logp(row, 1) > logp(row, 2)
            Yresult(row, 1) = 1;
         else
            Yresult(row, 1) = 0;
         end            
     end
      %====calculate the error====%       
    errorcounter = 0;
    for row = 1:Sx(1, 1)
        if Yresult(row, 1) ~= ytest(row, 1)
            errorcounter = errorcounter + 1;
        end
    end
    errortest(1,Ytestedcolumn) = errorcounter / 1536;
     %==print the Testing error rates for ¦Á = 1, 10 and 100
    if a == 1
         disp('error_rate_test_a=1: ')
        disp(errortest(1,Ytestedcolumn))
    end
    if a == 10
         disp('error_rate_test_a=10: ')
        disp(errortest(1,Ytestedcolumn))
    end
    if a == 100
         disp('error_rate_test_a=100: ')
        disp(errortest(1,Ytestedcolumn))
    end    
end
%========sketch the plots of test error rates versus a========%
figure(1);
scatter(0:0.5:100, errortest, 'r')
hold on;
figure(2);
plot(0:0.5:100, errortest, 'r')
hold on;

%% ===================For traing error=================%
Sx = size(BXtrain);
Ytestedcolumn = 0;
for a = 0:0.5:100
     thetajc = Thetajc(a);
     Ytestedcolumn = Ytestedcolumn + 1;
     logp = zeros(Sx(1,1) ,2);
     Yresult = zeros(Sx(1,1) ,1);
     %======probability of judging them to class 1======%
     for row = 1:Sx(1,1)
         mc1 = 0;
         for column = 1:Sx(1,2)
             xj = BXtrain(row, column);
             if xj == 1
                mc1 = mc1 + log(thetajc(1, column) );
             else
                mc1 = mc1 + log((1 - thetajc(1,column)) );
             end
             logp(row, 1) = log(paic1) + mc1;             
         end
     end         
     %======probability of judging them to class 0======%
     for row = 1:Sx(1,1)
         mc0 = 0;
         for column = 1:Sx(1,2)
             xj = BXtrain(row, column);
             if xj == 1
                mc0 = mc0 + log(thetajc(2, column) );
             else
                mc0 = mc0 + log((1 - thetajc(2, column)) );
             end
             logp(row, 2) = log(1-paic1) + mc0;             
         end 
     end     
     %======determine the class 0/1 by comparing======%
     for row = 1:Sx(1,1)
         if logp(row, 1) > logp(row, 2)
            Yresult(row, 1) = 1;
         else
            Yresult(row, 1) = 0;
         end            
     end
     %======calculate the error====%       
    errorcounter = 0;
    for row = 1:Sx(1, 1)
        if Yresult(row, 1) ~= ytrain(row, 1)
            errorcounter = errorcounter + 1;
        end
    end
    errortrain(1,Ytestedcolumn) = errorcounter / 3065;    
    %======print the Training error rates for ¦Á = 1, 10 and 100
    if a == 1
         disp('error_rate_train_a=1: ')
        disp(errortrain(1,Ytestedcolumn))
    end
    if a == 10
         disp('error_rate_train_a=10: ')
        disp(errortrain(1,Ytestedcolumn))
    end
    if a == 100
         disp('error_rate_train_a=100: ')
        disp(errortrain(1,Ytestedcolumn))
    end    
end
%========sketch the test error rates versus a========%
figure(1);
scatter(0:0.5:100, errortrain)
legend('test error rate', 'train error rate');  
title('Error rate -- Beta-bernoulli Naive Bayes');
figure(2);
plot(0:0.5:100, errortrain)
legend('test error rate', 'train error rate');  
title('Error rate -- Beta-bernoulli Naive Bayes');
%% ======Q2 Gaussian Naive Bayes ======%
%Use ML paic and maximum likelihood to estimate the class 
%conditional mean and variance of each feature
clc;
close all;
clear all;

%loading the mat file
load('spamData.mat');

%% ===========calculate paic1=p(y=c|T) using ML===========%
Nc1 = 0;
N = 3065;
%paic1
for row = 1:3065
    if ytrain(row,1) == 1
        Nc1 = Nc1 + 1;        
    end
end
paic1 = Nc1 / N;

%% =====================preprocessing features====================%
%=================Z-normalization=============% 
%==for train==%
ZXtrain = zscore(Xtrain);
%==for test==%    z-normalize the Xtest based on Xtrain
meantrain = mean(Xtrain);
stddevtrain = std(Xtrain);
for row = 1:1536
    for column = 1:57
        ZXtest(row, column) = (Xtest(row, column) - meantrain(1, column))/stddevtrain(1, column);
    end
end

%=================Log-transform===============%
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

%% ============devide a matrix to two matrixs============%%
%%=====devide the ZXtrain into two matrix(zmc1 and zmc0)=====%
rowcounter1 = 0;
rowcounter0 = 0;
for row = 1:3065
    if ytrain(row) == 1
        rowcounter1 = rowcounter1 + 1;
        zmc1(rowcounter1, :) = ZXtrain(row, :);

    else
        rowcounter0 = rowcounter0 + 1;
        zmc0(rowcounter0, :) = ZXtrain(row, :);
    end
end

%=====devide the LXtrain into two matrix(lmc1 and lmc0)=====%
lrowcounter1 = 0;
lrowcounter0 = 0;
for row = 1:3065
    if ytrain(row) == 1
        lrowcounter1 = lrowcounter1 + 1;
        lmc1(lrowcounter1, :) = LXtrain(row, :);

    else
        lrowcounter0 = lrowcounter0 + 1;
        lmc0(lrowcounter0, :) = LXtrain(row, :);
    end
end

%% =============calculate the error rate on Z-normalization ================%

               %==== Testing Z-normalization=====%
%calculate tvar1 and tvar0
tvar1 = std(zmc1).^2;
tvar0 = std(zmc0).^2;
%calculate tmean1 and tmean0
tmean1 = mean(zmc1);
tmean0 = mean(zmc0);

%calculate the probability of judging to class 1
sx = size(ZXtest);
for row = 1:sx(1,1)
    m1 = 0;
    for column = 1:sx(1,2)
        m1 = m1 + (-0.5*(ZXtest(row, column) - tmean1(1,column))^2)/tvar1(1, column)...
            + log(1/sqrt(2*pi*tvar1(1,column)));
        zlogp1(row,1) = log(paic1) + m1;
    end
end

%calculate the probability of judging to class 0
for row = 1:sx(1,1)
    m0 = 0;
    for column = 1:sx(1,2)
        m0 = m0 + (-0.5*(ZXtest(row, column) - tmean0(1,column))^2)/tvar0(1, column)...
            + log(1/sqrt(2*pi*tvar0(1,column)));
        zlogp0(row,1) = log(1-paic1) + m0;
    end
end

%determine the class 1 or 0 by comparing
for row = 1:sx(1,1)
    if zlogp1(row,1) > zlogp0(row,1)
        Yztest(row,1) = 1;
    else
        Yztest(row,1) = 0;
    end
end

%calculate the error rate for testing on Z-normalization
errorcounter = 0;
for row = 1:sx(1,1)
    if Yztest(row,1) ~= ytest(row,1)
        errorcounter = errorcounter + 1;
    end
end
error_rate_ztest = errorcounter / 1536;
disp('error_rate_Ztest :')
disp(error_rate_ztest)

           %==== Training Z-normalization=====%
%calculate tvar1 and tvar0
tvar1 = std(zmc1).^2;
tvar0 = std(zmc0).^2;
%calculate tmean1 and tmean0
tmean1 = mean(zmc1);
tmean0 = mean(zmc0);

%calculate the probability of judging to class 1
sx = size(ZXtrain);
for row = 1:sx(1,1)
    m1 = 0;
    for column = 1:sx(1,2)
        m1 = m1 + (-0.5*(ZXtrain(row, column) - tmean1(1,column))^2)/tvar1(1, column)...
            + log(1/sqrt(2*pi*tvar1(1,column)));
        zlogp1(row,1) = log(paic1) + m1;
    end
end

%calculate the probability of judging to class 0
for row = 1:sx(1,1)
    m0 = 0;
    for column = 1:sx(1,2)
        m0 = m0 + (-0.5*(ZXtrain(row, column) - tmean0(1,column))^2)/tvar0(1, column)...
            + log(1/sqrt(2*pi*tvar0(1,column)));
        zlogp0(row,1) = log(1-paic1) + m0;
    end
end

%determine the class 1 or 0 by comparing
for row = 1:sx(1,1)
    if zlogp1(row,1) > zlogp0(row,1)
        Yztest(row,1) = 1;
    else
        Yztest(row,1) = 0;
    end
end

%calculate the error rate for testing on Z-normalization
errorcounter = 0;
for row = 1:sx(1,1)
    if Yztest(row,1) ~= ytrain(row,1)
        errorcounter = errorcounter + 1;
    end
end
error_rate_ztrain = errorcounter / 3065;
disp('error_rate_Ztrain :')
disp(error_rate_ztrain)

%% =============calculate the error rate on Log-transform ================%

          %==== Testing Log-transform=====%
%calculate tvar1 and tvar0
tvar1 = std(lmc1).^2;
tvar0 = std(lmc0).^2;
%calculate tmean1 and tmean0
tmean1 = mean(lmc1);
tmean0 = mean(lmc0);

%calculate the probability of judging to class 1
sx = size(LXtest);
for row = 1:sx(1,1)
    m1 = 0;
    for column = 1:sx(1,2)
        m1 = m1 + (-0.5*(LXtest(row, column) - tmean1(1,column))^2)/tvar1(1, column)...
            + log(1/sqrt(2*pi*tvar1(1,column)));
        llogp1(row,1) = log(paic1) + m1;
    end
end

%calculate the probability of judging to class 0
for row = 1:sx(1,1)
    m0 = 0;
    for column = 1:sx(1,2)
        m0 = m0 + (-0.5*(LXtest(row, column) - tmean0(1,column))^2)/tvar0(1, column)...
            + log(1/sqrt(2*pi*tvar0(1,column)));
        llogp0(row,1) = log(1-paic1) + m0;
    end
end

%determine the class 1 or 0 by comparing
for row = 1:sx(1,1)
    if llogp1(row,1) > llogp0(row,1)
        Yltest(row,1) = 1;
    else
        Yltest(row,1) = 0;
    end
end

%calculate the error rate for testing on Z-normalization
errorcounter = 0;
for row = 1:sx(1,1)
    if Yltest(row,1) ~= ytest(row,1)
        errorcounter = errorcounter + 1;
    end
end
error_rate_ltest = errorcounter / 1536;
disp('error_rate_Ltest :')
disp(error_rate_ltest)

        %==== Training Log-transform=====%
%calculate tvar1 and tvar0
tvar1 = std(lmc1).^2;
tvar0 = std(lmc0).^2;
%calculate tmean1 and tmean0
tmean1 = mean(lmc1);
tmean0 = mean(lmc0);

%calculate the probability of judging to class 1
sx = size(LXtrain);
for row = 1:sx(1,1)
    m1 = 0;
    for column = 1:sx(1,2)
        m1 = m1 + (-0.5*(LXtrain(row, column) - tmean1(1,column))^2)/tvar1(1, column)...
            + log(1/sqrt(2*pi*tvar1(1,column)));
        llogp1(row,1) = log(paic1) + m1;
    end
end

%calculate the probability of judging to class 0
for row = 1:sx(1,1)
    m0 = 0;
    for column = 1:sx(1,2)
        m0 = m0 + (-0.5*(LXtrain(row, column) - tmean0(1,column))^2)/tvar0(1, column)...
            + log(1/sqrt(2*pi*tvar0(1,column)));
        llogp0(row,1) = log(1-paic1) + m0;
    end
end

%determine the class 1 or 0 by comparing
for row = 1:sx(1,1)
    if llogp1(row,1) > llogp0(row,1)
        Yltest(row,1) = 1;
    else
        Yltest(row,1) = 0;
    end
end

%calculate the error rate for testing on Z-normalization
errorcounter = 0;
for row = 1:sx(1,1)
    if Yltest(row,1) ~= ytrain(row,1)
        errorcounter = errorcounter + 1;
    end
end
error_rate_ltrain = errorcounter / 3065;
disp('error_rate_Ltrain :')
disp(error_rate_ltrain)

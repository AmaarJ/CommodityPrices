function [rmsvars,lowndx,rmstrain,rmstest] = a3_20103411
% [RMSVARS LOWNDX RMSTRAIN RMSTEST]=A3 finds the RMS errors of
% linear regression of the data in the file "MTLS.CSV" by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. For the variable that is
% best explained by the other variables, a 5-fold cross validation is
% computed. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         none
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing
[rmsvars,lowndx] = a3q1;
[rmstrain,rmstest] = a3q2(lowndx);
end

function [rmsvars,lowndx] = a3q1
% [RMSVARS LOWNDX]=A3Q1 finds the RMS errors of
% linear regression of the data in the file "MTLS.CSV" by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. 
%
% INPUTS:
%         none
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS

% Read the test data from a CSV file; remove first column and row
cPrices = csvread('mtls.csv', 1, 1);
%replace missing values with string NaN
cPrices(cPrices==0) = NaN;
%fills missing values with a moving mean
cP=fillmissing(cPrices,'movmean',20);
[m n] = size(cP);
%standardize the data
cP = zscore(cP);


% Compute the RMS errors for linear regression
Xmat = cP;
rmsvars = 1*(1:n);
%variable that would keep track of the lowest RMS value
minimum = 90000000000000000000000;
lowndx=100;
for i = 1:n
    %extract the i'th column to be the y vector
    B = Xmat(:, i);
    %remove the y vector from the data matrix
    cP(:,i) = [];
    A = cP;
    cP = Xmat;
    %solve the regression 
    X = linsolve(A,B);
    %calculate the error vector
    ErrorV = B-(A*X);
    RMS = sqrt(mean((ErrorV).^2));
    %update the lowndx and the minimum when a new minimum is calculated.
    if RMS<=minimum
        minimum = RMS;
        lowndx=i;
    end
    %return the rms value to the rmsvars array
    rmsvars(i) = RMS;
end

% Find the regression in unstandardized variables
cPrices = csvread('mtls.csv', 1, 1);
cPrices(cPrices==0) = NaN;
cP=fillmissing(cPrices,'movmean',20);
y1 = cP(:,lowndx);
cP(:,lowndx)= [];
x1 = cP;
X = linsolve(x1, y1);
y2 = x1*X;
% Plot the results
x2=linspace(1,112, 112);
plot(x2, y1, x2, y2);
title('Linear Regression of Best-Modeled Commodity');
xlabel('Quarterly index from 1992');
ylabel('Commodity price in USD');

fprintf('RMSVARS');
disp(rmsvars);
fprintf('LOWNDX:');
disp(lowndx);

end
function [rmstrain,rmstest] = a3q2(lowndx)
% [RMSTRAIN RMSTEST]=A3Q2(LOWNDX) finds the RMS errors of 5-fold
% cross-validation for the variable LOWNDX of the data in the file
% "MTLS.CSV" The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         LOWNDX   - integer scalar, index into the data
% OUTPUTS:
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

% Read the test data from a CSV file; find the size of the data
cPrices = csvread('mtls.csv', 1, 1);
cPrices(cPrices==0) = NaN;
cP=fillmissing(cPrices,'movmean',20);

% Create Xmat and yvec from the data and the input parameter,
dependent = cP(:,lowndx);
cP(:,lowndx)= [];
[m n] = size(cP);

% Compute the RMS errors of 5-fold cross-validation
P = cvpartition(m,'KFold',5);
rmstrain = 1*(1:5);
rmstest = 1*(1:5);
dep=dependent;
for i = 1:P.NumTestSets
    dependent=dep;
    %trIdx has the indices of the observations for the training set
    trIdx = P.training(i);
    %teIdx has the indices of the observations for the testing set
    teIdx = P.test(i);
    %use the tr indices to get the training set 
    trainSet = cP(trIdx,:);
    dependent = dependent(trIdx,:); 
    X=linsolve(trainSet, dependent);
    ErrorV = dependent-(trainSet*X);
    RMS = sqrt(mean((ErrorV).^2));
    rmstrain(i) = RMS;
    %use the te indices to get the testing set
    testSet = cP(teIdx,:);
    dependent=dep;
    dependent = dependent(teIdx,:); 
    %X=linsolve(testSet, dependent);
    ErrorV = dependent-(testSet*X);
    RMS = sqrt(mean((ErrorV).^2));
    rmstest(i) = RMS;
end
rmstrmean = mean(rmstrain);
rmstemean = mean(rmstest);
rmstrstd = std(rmstrain);
rmstestd = std(rmstest);
fprintf('RMSTRAIN:');
disp(rmstrain);
fprintf('RMSTEST:');
disp(rmstest);
fprintf('RMS TRAIN MEAN:');
disp(rmstrmean);
fprintf('RMS TRAIN STD DEV:');
disp(rmstrstd);
fprintf('RMS TEST MEAN:');
disp(rmstemean);
fprintf('RMS TEST STD DEV:');
disp(rmstestd);
end
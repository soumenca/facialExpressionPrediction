%Mallab version: R2017b

clc;
clear all;
close all;

data = csvread('/home/soumen/Desktop/TML_project/FERA-2013/originalFaceFeature_FERA-13.csv');

for x = 50:50:300
    fprintf(".......Output.......\n");
    reducedData = funPCA(data, x);
    
    funSVM(reducedData); % SVM Call
    funKNN(reducedData); % KNN Call
    funPNN(reducedData); % PNN Call
    
    [y, mse] = lmsFun(reducedData); % LMS Call
    [y1, mse1] = klmsFun(reducedData); % KLMS Call
    [y2, mse2] = mccFun(reducedData); % MCC Call
    [y3, mse3] = kmcFun(reducedData); % KMC Call
        
end

%% Classification .............LMS......

function [yd, mse] = lmsFun(reducedData) 
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .75, 0, .25);
    
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    
    %Training..............................
    d = trainClass';
    u = trainData';
    [M, N] = size(u);
    mu = 0.0085;
    w = zeros(1,M);
    for i = 1:N
        e(i) = d(i) - w * u(:,i);
        w = w + mu * e(i) * (u(:,i)');
        mse(i) = abs(e(i))^2;
    end
    yd=(w * u);
    
    %Testing................................
    d = testClass';
    u = testData';
    [M, N] = size(u);
    for i = 1:N
        e(i) = d(i) - w * u(:,i);
        w = w + mu * e(i) * (u(:,i)');
        mse(i) = abs(e(i))^2;
    end
    yd_test=(w * u);
    
    %Accuracy computation...................
    rangeY = min(yd_test):range(yd_test)/6:max(yd_test);
    yd_D = discretize(yd_test, rangeY);
    results = d == yd_D;
    true = sum(results == 1);
    accuracy = (true / length(d)) * 100;
    fprintf("The Accuracy of prediction using LMS is %0.4f\n", accuracy);
end

%% Classification .............MCC......

function [yd, mse] = mccFun(reducedData) 
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .75, 0, .25);
    
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    
    %Training..............................
    d = trainClass';
    u = trainData';
    [M, N] = size(u);
    mu = 0.001;
    sigma = 0.5;
    w = zeros(1,M);
    for i = 1:N
        e(i) = d(i) - w * u(:,i);
        w = w + mu * exp(- sigma * sum(e(i)).^2) * e(i) * (u(:,i)');
        mse(i) = abs(e(i))^2;
    end
    yd=(w * u);  
    
    %Testing................................
    d = testClass';
    u = testData';
    [M, N] = size(u);
    for i = 1:N
        e(i) = d(i) - w * u(:,i);
        w = w + mu * exp(- sigma * (e(i)).^2) * e(i) * (u(:,i)');
        mse(i) = abs(e(i))^2;
    end
    yd_test=(w * u);  
    
    %Accuracy computation...................
    rangeY = min(yd_test):range(yd_test)/6:max(yd_test);
    yd_D = discretize(yd_test, rangeY);
    results = d == yd_D;
    true = sum(results == 1);
    accuracy = (true / length(d)) * 100;
    fprintf("The Accuracy of prediction using MCC is %0.4f\n", accuracy);
end

%% Classification .............KLMS......

function [yd_test, mse] = klmsFun(reducedData) 
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .75, 0, .25);
    
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    
    %Training..............................
    d = trainClass';
    u = trainData';
    [M, N] = size(u);
    mu = 0.1;
    sigma = 0.004;
    e = d;
    y(1) = mu * e(1) * exp(- sigma * e(1) ^2); %exp((-e(1)^2) / (2*sigma^2));
    mse(1) = 0;
    for n = 2:N
        i = 1:(n-1);
        y(n) = mu * e(i)*(exp(- sigma * sum((u(:,i)-u(:,n)).^2)))';
        e(n) = d(n) - y(n);
        mse(n) = abs(e(n))^2;
    end  
    
    %Testing................................
    d = testClass';
    u = testData';
    [M, N] = size(u);
    y1(1) = mu * e(1) * exp(- sigma * e(1) ^2); %exp((-e(1)^2) / (2*sigma^2));
    for n = 2:N
        i = 1:(n-1);
        y1(n) = mu * e(i)*(exp(- sigma * sum((u(:,i)-u(:,n)).^2)))';
        e(n) = d(n) - y1(n);
        mse(n) = abs(e(n))^2;
    end  
    yd_test = y1;  
    
    %Accuracy computation...................
    rangeY = min(yd_test):range(yd_test)/6:max(yd_test);
    yd_D = discretize(yd_test, rangeY);
    results = d == yd_D;
    true = sum(results == 1);
    accuracy = (true / length(d)) * 100;
    fprintf("The Accuracy of prediction using KLMS is %0.4f\n", accuracy);
end

%% Classification .............KMC......

function [yd_test, mse] = kmcFun(reducedData) 
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .75, 0, .25);
    
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    
    %Training..............................
    d = trainClass';
    u = trainData';
    [M, N] = size(u);
    mu = 0.5;
    sigma = 0.0004;
    e = d;
    y(1) = mu * e(1) * exp((-e(1)^2) / (2*sigma^2));
    mse(1) = 0;
    for n = 2:N
        ii = 1:(n-1);
        y(n) = mu * e(ii).* exp(- sigma * (e(ii)).^2) * (exp(- sigma * sum((u(:,ii)-u(:,n)).^2)))';
        e(n) = d(n) - y(n);
        mse(n) = abs(e(n))^2;
    end  
    clear d;
    clear u;
    %Testing................................
    d = testClass';
    u = testData';
    [M, N] = size(u);
    y1(1) = mu * e(1) * exp((-e(1)^2) / (2*sigma^2));
    for n = 2:N
        ii = 1:(n-1);
        y1(n) = mu * e(ii).* exp(- sigma * (e(ii)).^2) * (exp(- sigma * sum((u(:,ii)-u(:,n)).^2)))';
        e(n) = d(n) - y1(n);
        mse(n) = abs(e(n))^2;
    end  
    yd_test = y1; 
    
    %Accuracy computation...................
    rangeY = min(yd_test):range(yd_test)/6:max(yd_test);
    yd_D = discretize(yd_test, rangeY);
    results = d == yd_D;
    true = sum(results == 1);
    accuracy = (true / length(d)) * 100;
    fprintf("The Accuracy of prediction using KMC is %0.4f\n", accuracy);
end

%% Classification ................SVM.........

function [] = funSVM(reducedData)
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .75, 0, .25);
    
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);
    valData = reducedData(valIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    valClass = reducedData(valIndex, noFeature);
    
    %t = templateSVM('Standardize',1,'KernelFunction','gaussian');
    %mdl = fitcecoc(trainData, trainClass, 'Learners', t);
    mdl = fitcecoc(trainData, trainClass);
    predictClass = predict(mdl, testData);
    results = testClass == predictClass;
    
    true = sum(results == 1);
    accuracy = (true / length(testClass)) * 100;
    fprintf("The Accuracy of prediction using SVM is %0.4f\n", accuracy);
    %cnfmat = confusionmat(testClass, predictClass)
end

%% Classification .............KNN.......

function [] = funKNN(reducedData)
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .75, 0, .25);
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);
    valData = reducedData(valIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    valClass = reducedData(valIndex, noFeature);

    Mdl = fitcknn(trainData, trainClass,'NumNeighbors',7);
    predictClass = predict(Mdl, testData);

    results = testClass == predictClass;
    true = sum(results == 1);
    accuracy = (true / length(testClass)) * 100;
    fprintf("The Accuracy of prediction using KNN is %0.4f\n", accuracy);
    
end

%% Classification .............PNN.......

function [] = funPNN(reducedData)
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .75, 0, .25);
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);
    valData = reducedData(valIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    valClass = reducedData(valIndex, noFeature);

    T = ind2vec(trainClass');
    net = newpnn(trainData',T);
    Y = net(testData');
    predictClass = vec2ind(Y);
    
    testClass = testClass';
    results = testClass == predictClass;
    true = sum(results == 1);
    accuracy = (true / length(testClass)) * 100;
    fprintf("The Accuracy of prediction using PNN is %0.4f\n", accuracy);
    
end



%% PCA Implementation .............

function reducedData = funPCA(feature, noFeature)
    [~, fc] = size(feature);
    feature1 = feature(:,1:(fc-1));
    %noFeature = 80;

    coeff= pca(feature1);
    reducedDimension = coeff(:, 1:noFeature);
    reducedData = feature1 * reducedDimension;
    reducedData = [reducedData, feature(:,fc)];
    fprintf("The feature of the original data is: %d\n",  fc-1);
    fprintf("The feature of the reduced data is: %d\n",  noFeature);
end
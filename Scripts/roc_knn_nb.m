% Clear workspace
clear all; clc; close all;

% Load the dataset and training/test sets
currentFolder = pwd;
dataPath = sprintf('%s/Data/X_test.csv', pwd);
X_test = readtable(dataPath);

dataPath = sprintf('%s/Data/X_train.csv', pwd);
X_train = readtable(dataPath);
    
dataPath = sprintf('%s/Data/y_test.csv', pwd);
y_test = readmatrix(dataPath);
    
dataPath = sprintf('%s/Data/y_train.csv', pwd);
y_train = readmatrix(dataPath);

% ROC performance KNN and NB
model = fitcknn(X_train, y_train,... 
        "NumNeighbors",23,...
        "Standardize",1,...
        "Distance","cityblock");
% TEST
[y_predict, scores] = predict(model, X_test);
performance(y_test, y_predict, scores(:,2))
hold on
model = fitcnb(X_train, y_train,'DistributionNames','kernel', 'Prior','uniform',...
            'Kernel', 'triangle','Width',0.001554);
% TEST
[y_predict, scores] = predict(model, X_test);
performance(y_test, y_predict, scores(:,2))
legend('KNN', 'NB')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Figure 11: ROC performance of KNN and NB')
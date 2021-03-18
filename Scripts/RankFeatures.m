%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INM431 Machine Learning Coursework %%
%% Zohra Bouchamaoui                  %%
%% Rank features for selection        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear workspace
clear all; clc; close all;

% Load the dataset
currentFolder = pwd;
dataPath_clean = sprintf('%s/Data/adult_clean.csv', pwd);
df = readtable(dataPath_clean, 'ReadVariableNames', true);

% Data Shape
[m n] = size(df); % m = number of rows ; n = number of columns

% Rank the predictors in the dataset while using the 'income' column as the
% target variable
[apx,scores] = fscmrmr(df, 'income', 'Weight', 'fnlwgt') % The MathWorks (2020)

% Plot bar chart to see the ranking of the variables
bar(scores(apx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
xticklabels(strrep(df.Properties.VariableNames(apx),'_','\_'))
xtickangle(45)
title('Figure 9: Ranking of features')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INM431 Machine Learning Coursework %%
%% Zohra Bouchamaoui                  %%
%% Data Cleaning                      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear workspace
clear all; clc; close all;

% Load the dataset
currentFolder = pwd;
dataPath = sprintf('%s/Data/adult.csv', pwd);
dataPath_clean = sprintf('%s/Data/adult_clean.csv', pwd);
df = readtable(dataPath);

% Data Shape
[m n] = size(df); % m = number of rows ; n = number of columns - The MathWorks (2020)

% Data Pre-processing : Transforming categorical variables into the categorical type
Categoricalcolumns = {'workclass', 'education', 'marital_status', 'occupation','relationship', 'race', 'gender', 'native_country', 'income'};
CategoricalcolumnsFilter = ismember(df.Properties.VariableNames, Categoricalcolumns); % The MathWorks (2020)
for i = 1:length(Categoricalcolumns)
    column = Categoricalcolumns{i};
    df.(column) = categorical(df{:, column});
end

% Print Table Summary
fprintf('------------------------------------------------------------------\n')
fprintf('The dataset has %d Rows and %d columns.\n\n', m, n)
fprintf('The dataset has %d Numeric and %d Categorical attributes.\n',...
    n-length(Categoricalcolumns), length(Categoricalcolumns))
fprintf('------------------------------------------------------------------\n')
summary(df)

% Find missing values and remove all the rows containing them
sum(ismissing(df, {'?'})) % Sum of the missing values - The MathWorks, (2020)

% Remove rows with missing entries 
Todel = df.workclass == '?';
df(Todel, :) =[];
Todel1 = df.occupation == '?';
df(Todel1, :) =[];
Todel2 = df.native_country == '?';
df(Todel2, :) =[];

sum(ismissing(df, {'?'})) % Sum of the missing values - The MathWorks (2020)
% The original total of rows for this dataset is 48,842 but after removing
% the empty rows, we are left with 45,222 rows

% Drop the education column as it is redundant with the educational_num
% column
df(:, 4) = [];

% Data Shape after cleaning
[m n] = size(df); % m = number of rows ; n = number of columns - The MathWorks (2020)

% Save the cleaned dataset into a new csv file
writetable(df,dataPath_clean)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INM431 Machine Learning Coursework %%
%% Zohra Bouchamaoui                  %%
%% Exploratory Analysis               %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear workspace
clear all; clc; close all;

% Load the dataset
currentFolder = pwd;
dataPath_clean = sprintf('%s/Data/adult_clean.csv', pwd);
df = readtable(dataPath_clean, 'ReadVariableNames', true);

% Data Shape
[m n] = size(df); % m = number of rows ; n = number of columns - The MathWorks (2020)

% Data Pre-processing : Transforming categorical variables into the categorical type
Categoricalcolumns = {'workclass', 'marital_status', 'occupation','relationship', 'race', 'gender', 'native_country', 'income'};
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

% Check for class imbalances using the target (income column)
high_income = df(df.income == '>50K',:); 
low_income = df(df.income == '<=50K',:);

% Representing the class imbalances as percentages
high_income_per = (size(high_income)/size(df.income))*100; % 24.78%
low_income_per = (size(low_income)/size(df.income))*100; % 75.22%

% Determining the numerical feature variables
features_num = {'age', 'educational_num', 'capital_gain', 'capital_loss','hours_per_week'};
df_num = df(:,features_num);

% Finding high_income and low_income using only numerical variables
high_income_num = df_num(df.income == '>50K',:); 
low_income_num = df_num(df.income == '<=50K',:);

% Descriptive statistics
% Calculating the mean of the numerical columns
% When income is >50K
mean_above = mean(table2array(high_income_num), 1)
% When income is <=50K
mean_below = mean(table2array(low_income_num), 1)

% Calculating the standard deviation of the numerical columns
% When income is >50K
std_above = std(table2array(high_income_num), 1)
% When income is <=50K
std_below = std(table2array(low_income_num), 1)

% Calculating the skewness of the numerical columns
% When income is >50K
skewness_above = skewness(table2array(high_income_num), 1)
% When income is <=50K
skewness_below = skewness(table2array(low_income_num), 1)

% Plots of the numerical variables with the two income classes
% Figure for age when income '>50K' and '<=50K'
 figure('pos',[450 10 500 400])
 histogram(high_income.age)
 hold on
 histogram(low_income.age)
 legend('income >50K', 'income <=50K')
 xlabel('Age')
 ylabel('number of individuals')
 title('Figure 4: Age of the individuals when income >50K and <=50K')

% Figure for number of education years when income '>50K' and '<=50K' 
 figure('pos',[450 10 500 400])
 histogram(high_income.educational_num)
 hold on
 histogram(low_income.educational_num)
 legend('income >50K', 'income <=50K')
 xlabel('Number of education years')
 ylabel('number of individuals')
 title('Figure 5: Number of education years of the individuals when income >50K and <=50K')
 
% Figure for capital gain when income '>50K' and '<=50K' 
 figure('pos',[450 10 500 400])
 subplot(2,1,1)
 boxplot(high_income.capital_gain)
 title('Figure 7: Capital Gain of the individuals when income >50K ')
 subplot(2,1,2)
 boxplot(low_income.capital_gain)
 title('Capital Gain of the individuals when income <=50K')

% Figure for capital loss when income '>50K' and '<=50K' 
 figure('pos',[450 10 500 400])
 subplot(2,1,1)
 boxplot(high_income.capital_loss)
 title('Figure 8: Capital Loss of the individuals when income >50K ')
 subplot(2,1,2)
 boxplot(low_income.capital_loss)
 title('Capital Loss of the individuals when income <=50K')
 
% Figure for capital gain when income '>50K' and '<=50K' 
 figure('pos',[450 10 500 400])
 histogram(high_income.capital_gain)
 hold on
 histogram(low_income.capital_gain)
 legend('income >50K', 'income <=50K')
 xlabel('Capital gain')
 ylabel('number of individuals')
 title('Capital Gain of the individuals when income >50K and <=50K')

% Figure for capital loss when income '>50K' and '<=50K' 
 figure('pos',[450 10 500 400])
 histogram(high_income.capital_loss)
 hold on
 histogram(low_income.capital_loss)
 legend('income >50K', 'income <=50K')
 xlabel('Capital Loss')
 ylabel('number of individuals')
 title('Capital Loss of the individuals when income >50K and <=50K')
 
% Figure for hours per week when income '>50K' and '<=50K' 
 figure('pos',[450 10 500 400])
 histogram(high_income.hours_per_week)
 hold on
 histogram(low_income.hours_per_week) 
 legend('income >50K', 'income <=50K')
 xlabel('Number of hours per week')
 ylabel('number of individuals')
 title('Figure 6: Hours per week of the individuals when income >50K and <=50K')
 
% Univariate Analysis 
% Plot the histograms for each individual variable
 figure('pos',[50 50 2800 2000])
 for index_column = 1:n
     subplot(4,4,index_column)
     histogram(df{:, index_column})
     title(sprintf('Histogram representation of %s', df.Properties.VariableNames{index_column}))
 end
 
% Plot the class imbalance using the target variable 'income'
 figure('pos',[10 1000 500 400])
 histogram(df.income)
 title('Figure 3: Target Variable - Class Imbalance')

% Categorical and numerical features: we transformed categorical features
% into numerical variables
% to plot the Correlation Matrix
features_cat = {'workclass', 'marital_status', 'occupation','relationship', 'race', 'gender', 'native_country', 'income'};
features_num = {'age', 'fnlwgt','educational_num', 'capital_gain', 'capital_loss', 'hours_per_week'}
df_cat = df(:,features_cat);
df_num = df(:,features_num);
df = [df_num array2table(double(table2array(df_cat)), 'VariableName', features_cat)];
    
% Bivariate Analysis 
     % Compute Correlation Matrix (Pearson coefficient)
 CorrColumns = {'age', 'workclass', 'fnlwgt', 'educational_num', 'marital_status', 'occupation' 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'income'}; % I used the numerical columns for this analysis
 CorrColumnsFilter = ismember(df.Properties.VariableNames, CorrColumns); % (Mathworks (2020). 'ismember'. [Online] Available at: https://www.mathworks.com/help/matlab/ref/double.ismember.html)
 corrMatrix = corr(table2array(df(:, CorrColumns)),'type','Pearson'); % (Mathworks (2020). 'corr'. [Online] Available at: https://www.mathworks.com/help/stats/corr.html?searchHighlight=corr&s_tid=srchtitle)

     % Plot the Correlation Matrix
 figure('pos',[1000 1000 500 400])
 labels = df.Properties.VariableNames(CorrColumnsFilter); % Label of the numeric attributes to add to the matrix
 colormap('cool') % colormap theme - The MathWorks (2020)
 color_limts = [-1, 1]; % colormap scale limits - The MathWorks (2020)
 imagesc(corrMatrix, color_limts)  % Visualize - The MathWorks (2020)
 colorbar % Show colorbar - The Mathworks (2020)
 
     % Add Text (values) to Matrix
 text_values = num2str(corrMatrix(:), '%0.2f'); 
 text_values = strtrim(cellstr(text_values));
 
     %  Use a and b as coordinates for the text values on the matrix
 [a, b] = meshgrid(1:14); 
 hStrings = text(a(:), b(:), text_values(:), 'HorizontalAlignment', 'center');
 set(hStrings, 'color', 'black') % Set color of text
 
     % Add the Axis Labels, centre the Axis ticks on the bins and add a Title
     % a-axis labels
 set(gca, 'XTickLabel', labels);
     % b-axis labels
 set(gca, 'YTickLabel', labels);
      % centre a-axis ticks
 set(gca, 'XTick', 1:14); 
      % centre b-axis ticks
 set(gca, 'YTick', 1:14); 
      % Title
 title(' Figure 1: Pearson Correlation Matrix')

 % Multivariate Analysis : Parallel Coordinates
     % Scaling Numeric columns
CorrCols = {'age','fnlwgt', 'educational_num', 'capital_gain', 'capital_loss', 'hours_per_week'}; % I used the numerical columns for this analysis 
Numeric_cols = table2array(df(:, CorrCols)); % Table to Matrix
mean_data = mean(Numeric_cols); std_data = std(Numeric_cols);
Numeric_colsNorm = (Numeric_cols - mean_data)./std_data; % Normalize
 
     % Scatter Plot Matrix
figure
gplotmatrix(Numeric_cols,[],df.income,['b' 'r'],[],[],false); 
text([.08 .22 .35 .55 .72 .86], repmat(-.1,1,6), CorrCols, 'FontSize',8);
text(repmat(-.1,1,6), [.88 .72 .52 .38 .22 .04], CorrCols, 'FontSize',8, 'Rotation',90); % The MathWorks (2020)
title(' Figure 2: Multidimensional scatter plot of numerical variables')
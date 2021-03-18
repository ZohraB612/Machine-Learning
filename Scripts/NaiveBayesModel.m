%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INM431 Machine Learning Coursework %%
%% Zohra Bouchamaoui                  %%
%% K-Nearest Neighbors                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear workspace
clear all; clc; close all;

% Set random seed to be able to repeat same results
rng(1)

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
   
% MODEL STARTS
mode = 0
if mode == 0 % optimize
    %Grid search to find the best hyperparameters
    model = fitcnb(X_train,y_train, "OptimizeHyperparameters",'all',...
        "HyperparameterOptimizationOptions", struct("AcquisitionFunctionName",'expected-improvement-plus'));
end
if mode == 1
  % train/validate/test
      fprintf("training..")
        model = fitcnb(X_train, y_train,'DistributionNames','kernel', 'Prior','uniform',...
            'Kernel', 'triangle','Width',0.001554);
    
    save("best_nb.mat", "model")
    load("best_nb.mat","model") 
    
    %TRAIN
    [y_train_predict, train_scores] = predict(model, X_train);
    performance(y_train, y_train_predict, train_scores(:,2))
    
    % VALIDATION 
    cv_model = crossval(model,'KFold' , 10)
    %cat("Cross validating...")
    loss = kfoldLoss(cv_model,'lossfun','classiferror' )
    accuracy = 1 - loss
    
    % TEST
    [y_predict, scores] = predict(model, X_test);
    performance(y_test, y_predict, scores(:,2))
    
 end 
% feature selection
if mode == 2
    n_features = width(X_train);
    imp_features = {'occupation','gender', 'capital_gain', 'fnlwgt', 'capital_loss','relationship', 'age','marital_status','educational_num','workclass','race' 'hours_per_week', 'native_country'};
    scores = [];
    
    cv = cvpartition(y_train, 'HoldOut', 0.3);
    test_indice = cv.test;
    
    X_train = X_train(:,imp_features);
    X_train2 = X_train(~test_indice,:);
    X_valid = X_train(test_indice,:);

    y_train2 = y_train(~test_indice,:);
    y_valid = y_train(test_indice,:);
    
    for i=2:n_features
        X_train_sub = X_train2(:,1:i);
        X_valid_sub = X_valid(:,1:i);
        
        model = fitcnb(X_train_sub, y_train2,'DistributionNames','kernel', 'Prior','uniform',...
            'Kernel', 'triangle','Width',0.001554);
        
        [y_predict, valid_scores] = predict(model, X_valid_sub);
       
        cm = confusionmat(y_valid, y_predict);
        TP = cm(1,1);
        TN = cm(2,2);
        FP = cm(1,2);
        FN = cm(2,1);
    
        n = TP + FN + TN + FP;
        accuracy = (TP + TN)/ n;
        scores = [scores accuracy]
        
    end  
end
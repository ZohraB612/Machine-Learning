function feature_data_split()
    currentFolder = pwd;
    dataPath = sprintf('%s/Data/adult_clean.csv', pwd);
    df = readtable(dataPath);
    
    df.income =  int8(strcmp(df.income,  '>50K'));

    
    % Data Shape
    [m n] = size(df); % m = number of rows ; n = number of columns

    % Data Pre-processing : Transforming categorical variables into the categorical type
    Categoricalcolumns = {'workclass', 'marital_status', 'occupation','relationship', 'race', 'gender', 'native_country' };
    CategoricalcolumnsFilter = ismember(df.Properties.VariableNames, Categoricalcolumns); % https://towardsdatascience.com/encoding-categorical-features-21a2651a065c
    for i = 1:length(Categoricalcolumns)
        column = Categoricalcolumns{i};
        df.(column) = categorical(df{:, column});
    end

    % Split the data into training and test set 70/30 using partition data for
    % cross-validation
    rng(1) %for reproducibility
    cv = cvpartition(df.income, 'HoldOut', 0.3); % The MathWorks (2020)
    test_indice = cv.test;

    % Determining the categorical and numerical feature variables
    features_cat = {'workclass', 'marital_status', 'occupation','relationship', 'race', 'gender', 'native_country'};
    features_num = {'age', 'fnlwgt', 'educational_num', 'capital_gain', 'capital_loss', 'hours_per_week'}
    
    df_num = df(:,features_num);

    % normilize num
    df_num = normalize(df_num);
    
    % We encoded the categorical data into one-hot vectors
    X = [];
    for i=1:width(df)
        if CategoricalcolumnsFilter(i)==1 % if Categorical Feature
            numericalEncoding = grp2idx(df.(i));
            X = [X numericalEncoding];
        end
    end

    % Determining the feature variables and the target variable
    df_cat = array2table(X, 'VariableName',features_cat);
    X = [df_cat df_num];
    y = df.income; % int8(df.income ==  '>50K');

    % Split the data intro training and test sets
    X_train = X(~test_indice,:);
    X_test = X(test_indice,:);

    y_train = y(~test_indice,:);
    y_test = y(test_indice,:);

    % Save the training and test sets as csv files to be able to load them
    % easier in other MATLAB files
    dataPath = sprintf('%s/Data/X_test.csv', pwd);
    writetable(X_test, dataPath);
    
    dataPath = sprintf('%s/Data/X_train.csv', pwd);
    writetable(X_train, dataPath);
    
    dataPath = sprintf('%s/Data/y_test.csv', pwd);
    writematrix(y_test, dataPath);
    
    dataPath = sprintf('%s/Data/y_train.csv', pwd);
    writematrix(y_train, dataPath);
    
end
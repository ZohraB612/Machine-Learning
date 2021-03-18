# Machine-Learning

# Introduction

This ReadMe file contains instructions and remarks concerning the coursework deliverables developed by Zohra Bouchamaoui. 

# Matlab Version

All the work has been developed using Matlab R2020a (9.8.0.1451342).

# Deliverables Structure

In the present folder, you will find :
	- One folder containing the dataset (raw and cleaned version) as well as the training and test sets: 	X_test.csv, X_train.csv, y_test.csv and y_train.csv
	- One folder containing this ReadMe file (text file), the poster (pdf file) and additional information (pdf)
	- Eleven MATLAB files (nine from which are .m and the two best trained models .mat)

# MATLAB files

In this section, I explain how the scripts were developed. All scripts are meant to run independently. The MATLAB scripts are designed to run from within the ‘Scripts’ folder using the pwd function in MATLAB. The dataset is accessible using pwd/Data and the user will be able to execute the scripts and use the dataset as long as the deliverables are not taken out of the main folder.

	- DataCleaningAdult.m: This script presents the pre-processing of the dataset where first, the categorical variables were transformed into the categorical type, then I removed the rows with missing entries and dropped the ‘education’ column as it seemed that its information was redundant with the one of the ‘educational_num’ column. The cleaned/pre-processed dataset was saved as adult_clean.csv and will be the data used for all the following scripts.

	- ExploratoryAnalysisAdult.m: This script presents some descriptive statistics about the Adult dataset with first a general overview given by a summary table and then a Univariate, Bivariate and Multivariate analysis.

	- RankFeatures.m : This script runs a minimum redundancy maximum relevance (based on mutual information) to perform feature ranking. It shows the ranking of the data’s predictors from the most important to the least important one.

	- feature_Selection.m : This script plots  the models’ scores every time we feed the model a new feature. The score results for K-NN and NB come from the KNNModel.m and NaiveBayesModel.m files.

	- performance.m : This script presents a performance function which calculates the measures necessary to compare our models. Measures calculated are: AUC, accuracy, f1_score, precision, recall, and fallout, using the True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN) values. This function will be used in both the KNNModel.m and NaiveBayesModel.m files/

	- feature_data_split.m : This script presents the split of the dataset into 70/30% using train/test cross-validation. It also contains the hot-encoding of the categorical variables to transform them into numerical values. We then create X_train, X_test, y_train and y_test which are saved to allow easier use in other .m files.  

	- KNNModel.m : This script runs the KNN Model. First we loaded X_train, X_test, y_train and y_test. Then we started training the model. We set a random seed for reproducibility purposes. 
To optimise the model using Bayesian Optimisation set mode = 0. 
To train, validate and test the model set mode = 1. (This can take around 27-28 seconds to run)
To run feature selection set mode = 2.

	- NaiveBayesModel.m: This script runs the Naive Bayes Model. First we loaded X_train, X_test, y_train and y_test. Then we started training the model. We set a random seed for reproducibility purposes. 
To optimise the model using Bayesian Optimisation set mode = 0. 
To train, validate and test the model set mode = 1. (WARNING - This can take around 5 minutes to run)
To run feature selection set mode = 2.

	- roc_knn_nb.m: This script runs the test sets for both models and plots their ROC curves and prints their performance measures.

	- best_knn.mat: This file stores the best trained KNN model and is used in the KNNModel.m file. 

	- best_nb.mat: This file stores the best trained NB model and is used in the KNNModel.m file. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INM431 Machine Learning Coursework %%
%% Zohra Bouchamaoui                  %%
%% Features selection                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Feature selection
scores_knn= [  0.7797  ,  0.8004   , 0.7992  ,  0.8064 ,   0.8318  ,  0.8379  ,  0.8365 ,   0.8358 ,   0.8410   , 0.8405  ,  0.8400  ,  0.8398];
scores_nb = [   0.6631  ,  0.6973  ,  0.7149 ,   0.7409  ,  0.7605 ,   0.7726  ,  0.7546 ,   0.7771  ,  0.7794 ,   0.7815   , 0.7822  ,  0.7848];

plot (1:12, scores_knn)
hold on
plot(1:12, scores_nb)
legend('KNN ', 'NB')
xlabel('features')
ylabel('scores')
title('Figure 10: Score plot for KNN and NB according to features')


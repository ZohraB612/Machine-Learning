function  performance(y_test, y_predict, scores)
   [Xpr,Ypr,Tpr, AUCpr] = perfcurve(y_test, scores,1); % MathWorks (2020)
    plot(Xpr, Ypr);
    print("AUC");
    AUCpr 
    
    % conf matrix
    cm = confusionmat(y_test, y_predict); % MathWorks (2020)
    TP = cm(1,1)
    TN = cm(2,2)
    FP = cm(1,2)
    FN = cm(2,1)
    
    n = TP + FN + TN + FP
    accuracy = (TP + TN)/ n
    f1_score= 2 * TP/ ( 2 * TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP/ n
    fallout = FP/(FP + TN)
    
end
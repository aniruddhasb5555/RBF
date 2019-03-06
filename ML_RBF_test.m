function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,macrof1,microf1,precision, Recall, FMeasure, Accuracy,SubsetAccu,Outputs,Pre_Labels,te_time]=ML_RBF_test(test_data,test_target,Centroids,Sigma_value,Weights)
%ML_RBF_test tests a multi-label RBF learner as in [1]
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels,te_time]=ML_RBF_test(test_data,test_target,Centroids,Sigma_value,Weights)
%
%    Description
%
%       ML_RBF_test takes,
%           test_data        - An MxN array, the i-th testing instance is stored in test_data(i,:)
%           test_target      - A QxM array, if the i-th testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           Centroids        - A KxN matrix, where the k-th centroid of the RBF neural network is stored in Centroids(k,:)
%           Sigma_value      - A 1xK vector, where the sigma value for the k-th centroid is stored in Sigma_value(1,k)
%           Weights          - A (K+1)xQ matrix used for label prediction
%      and returns,
%           HammingLoss       - The hamming loss on testing data
%           RankingLoss       - The ranking loss on testing data
%           OneError          - The one-error on testing data
%           Coverage          - The coverage on testing data
%           Average_Precision - The average precision on testing data
%           Outputs           - A QxM array, the probability of the i-th testing instance belonging to the j-th class is stored in Outputs(j,i)
%           Pre_Labels        - A QxM array, if the i-th testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%           te_time           - The time spent in testing
%
% [1] M.-L. Zhang. ML-RBF: RBF neural networks for multi-label learning. Neural Processing Letters, 2009, 29(2): 61-74.
   
    start_time=cputime;
    
    [num_class,num_test]=size(test_target);
    num_centroid=size(Centroids,1);
    num_test=size(test_data,1);
    A=zeros(num_test,num_centroid);
    
    for i=1:num_test
        for j=1:num_centroid
            A(i,j)=exp((-1)*(sum((test_data(i,:)-Centroids(j,:)).^2 )./ (2*Sigma_value(j)*Sigma_value(j))));
        end
    end
    A=[ones(num_test,1) A];    
    Outputs=(A*Weights)';
    
%Evaluation
    Pre_Labels=zeros(num_class,num_test);
    for i=1:num_test
        for j=1:num_class
            if(Outputs(j,i)>=0)
                Pre_Labels(j,i)=1;
            else
                Pre_Labels(j,i)=-1;
            end
        end
    end
    HammingLoss=Hamming_loss(Pre_Labels,test_target);
    RankingLoss=Ranking_loss(Outputs,test_target);
    OneError=One_error(Outputs,test_target);
    Coverage=coverage(Outputs,test_target);
    Average_Precision=Average_precision(Outputs,test_target);
    macrof1 = MacroF1(Pre_Labels,test_target);
    microf1 = MicroF1(Pre_Labels,test_target);
    [precision, Recall, FMeasure, Accuracy]=Precision(Pre_Labels,test_target);
    SubsetAccu=Subset_Accuracy(Pre_Labels,test_target);
    te_time=cputime-start_time;
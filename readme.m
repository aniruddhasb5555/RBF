%This is an examplar file on how the ML-RBF program could be used (The main function is "ML_RBF_train.m" and "ML_RBF_test.m")
%
%Type 'help ML_RBF_train' and 'help ML_RBF_test' under Matlab prompt for more detailed information


% Loading the file containing the necessary inputs for calling the ML-RBF function
addpath('C:\Users\aniru\Desktop\ML\Multi-label\Mat data'); 
%load('yeast_TTdata');
%no_of_class = 14;
%Filename
%Corel5k_data.mat
%Birds_data.mat
%Scene_data.mat
%Yeast_data.mat

%............................PREPARATION of DATA...........................
%Concat Train test
[Sample, Sample_label, no_of_class, fp]=read_file();
load(fp);
[no_of_sample, column] = size(Sample);
% Idx=randperm(no_of_sample);
% Sample1=Sample(Idx,:);

%genbase
% Sample=Sample(:,2:column);
% column=column-1;
no_of_feature = column - no_of_class;

train_data=Sample(:,1:no_of_feature);
train_target=Sample(:,no_of_feature+1:end);
train_target=train_target';
%test_target=test_data(:,no_of_feature+1:end);
%test_data=test_data(:,1:no_of_feature);
%test_target=test_target';
train_target(train_target==0)=-1;
%test_target(test_target==0)=-1;
%Set parameters for the ML-RBF algorithm
ratio=0.01;% Set the alpha parameter
mu=1; % Set the mu parameter
k=5;%NO_of_folds for testing data
DD=Kfoldcreat(Sample,no_of_class,k);
% Calling the main functions
[Centroids,Sigma_value,Weights,tr_time]=ML_RBF_train(train_data,train_target,ratio,mu); % Invoking the training procedure
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,macrof1,microf1,precision, Recall, FMeasure, Accuracy,SubsetAccu,Outputs,Pre_Labels,te_time]=ML_RBF_test(train_data,train_target,Centroids,Sigma_value,Weights);
fprintf('Note the results\n');
mainResult=[no_of_sample,no_of_feature,no_of_class,Average_Precision,HammingLoss,RankingLoss,OneError,Coverage,macrof1,microf1,precision, Recall, FMeasure, Accuracy,SubsetAccu ,tr_time];
result=cell(1,1);
result{1}=[];
for i=1:k
[Centroids,Sigma_value,Weights,tr_time]=ML_RBF_train(DD{i,1},DD{i,2},ratio,mu); % Invoking the training procedure
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,macrof1,microf1,precision, Recall, FMeasure, Accuracy,SubsetAccu,Outputs,Pre_Labels,te_time]=ML_RBF_test(DD{i,3},DD{i,4},Centroids,Sigma_value,Weights);
%[Centroids,Sigma_value,Weights,tr_time]=ML_RBF_train(DD{i,1},DD{i,2},ratio,mu); % Invoking the training procedure
%[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels,te_time]=ML_RBF_test(DD{i,3},DD{i,4},Centroids,Sigma_value,Weights); % Performing the test procedure
%E=errorcal(train_data,train_target,Centroids,Sigma_value,Weights);
result{1}=[result{1};no_of_sample,no_of_feature,no_of_class,Average_Precision,HammingLoss,RankingLoss,OneError,Coverage,macrof1,microf1,precision, Recall, FMeasure, Accuracy,SubsetAccu ,tr_time];
fprintf('Note the results of %dth fold\n',i);
end
fprintf('Done\n');
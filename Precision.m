function [Precision, Recall, FMeasure, Accuracy]=Precision(Pre_Labels,test_target)
%test_target: original
%Pre_Labels: predicted
%Assume labels are 0 and 1
Pre_Labels(Pre_Labels==-1)=0;
test_target(test_target==-1)=0;
%     [no_of_test,num_class]=size(Pre_Labels);

%     temp_Outputs=[];
%     temp_test_target=[];
%     for i=1:no_of_test
%         temp=test_target(i,:);
%         if((sum(temp)~=num_class)&&(sum(temp)~=0))%AL
%             temp_Outputs=[temp_Outputs,Pre_Labels(i,:)];
%             temp_test_target=[temp_test_target,temp];
%         end
%     end
    
%     Pre_Labels=temp_Outputs;
%     test_target=temp_test_target;     
    [no_of_test,~]=size(Pre_Labels);

    temp1=Pre_Labels.*test_target;
    temp=sum(temp1,2);

    temp_sum=sum(test_target,2);
    temp_sum1=sum(Pre_Labels,2);
    temp_sum2=temp_sum+temp_sum1;

    tmp=temp./temp_sum;
    tmp(isnan(tmp)==1) = 0;
    Precision=sum(tmp)/no_of_test;
    
    tmp1=temp./temp_sum1;
    tmp1(isnan(tmp1)==1) = 0;
    Recall=sum(tmp1)/no_of_test;
    
    tmp2=(2*temp)./temp_sum2;
    tmp2(isnan(tmp2)==1) = 0;
    FMeasure=sum(tmp2)/no_of_test;

    temp_sum31=Pre_Labels+test_target;
    temp_sum31(temp_sum31==2)=1;
    temp_sum3=sum(temp_sum31,2);
    tmp3=temp./temp_sum3;
    tmp3(isnan(tmp3)==1) = 0;
    Accuracy=sum(tmp3)/no_of_test;
    
end
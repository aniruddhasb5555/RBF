function [Result] = Performance_measure(Test_Final,Test_label,Test_Output)

%Test_label = Actual
%Test_Final = Predicted
%Test_Output = before hard labelling

    [no_of_pattern,no_of_class] = size(Test_Final);
    Exact_Accuracy=0;
    count = 0;
    %count1 = 0;
    one_error_test = 0;
    [max_test,pos] = max(Test_Output,[],2);
    coverage_sum=0;
    AP_sum = 0;
    
    
    for i = 1:no_of_pattern
        flag = 0;
        Test_temp = Test_Output(i,:);
        [Temp1,I] = sort(Test_temp);
        
        %Coverage
        for j = no_of_class:-1:1
            temp2 = I(j);
            if Test_label(temp2)==1
                break;
            end
        end
         coverage_sum = coverage_sum + temp2;
         
        %Average precision
        AP_count = nnz(find(Test_label(i,:)==1));
         
         for j = 1:no_of_class
           if Test_label(i,j) ~= Test_Final(i,j)
              flag = flag+1;
           end
         end
        count = count + flag;
                
        if flag == 0
            Exact_Accuracy = Exact_Accuracy+1;
        end
        if Test_Final(i,pos(i)) ~= Test_label(i,pos(i))
            one_error_test = one_error_test+1;
        end    

    end

    %Exact match Accuracy
    E_A=(Exact_Accuracy/no_of_pattern)*100;
    
    %Hamming Loss
    H_Loss = (double(count)/(no_of_pattern*no_of_class))*100;
    %H_Loss_1 = (double(count1)/(no_of_pattern*no_of_class))*100;
    
    %One Error
    One_error=(double(one_error_test)/no_of_pattern)*100;
    
    %Coverage
    Coverage = (double(coverage_sum)/no_of_pattern)-1;
    
    Result(1) = E_A;
    Result(2) = H_Loss;
    Result(3) = 100-H_Loss;
    Result(4) = One_error;
    Result(5) = Coverage;
    Result(6) = 
   
end
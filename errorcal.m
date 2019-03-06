function E = errorcal(test_data,test_target,Centroids,Sigma_value,Weights)
[num_class, num_test ]=size(test_target);
    num_centroid=size(Centroids,1);
    num_test=size(test_data,1);
    A=zeros(num_test,num_centroid);
    
    for i=1:num_test
        for j=1:num_centroid
            A(i,j)=exp((-1)*(sum((test_data(i,:)-Centroids(j,:)).^2 )./ (Sigma_value(j)*Sigma_value(j))))/(sqrt(pi)*Sigma_value(j));
        end
    end
    A=[ones(num_test,1) A];    
    Outputs=(A*Weights)';

    for i=1:num_test
        for j=1:num_class
            if(Outputs(j,i)>=0)
                Outputs(j,i)=1;
            else
                Outputs(j,i)=-1;
            end
        end
    end
    E=sum(sum(abs(Outputs-test_target)))/(num_test);
end
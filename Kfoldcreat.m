function train=Kfoldcreat(Sample,no_of_class,k)
    [no_of_sample, column] = size(Sample);
    no_of_feature = column - no_of_class;
    Indices = crossvalind('Kfold', no_of_sample, k);
    train=cell(k,4);
    for i=1:k
        train_data=zeros(length(Indices(Indices~=k)),no_of_feature);
        train_target=zeros(length(Indices(Indices~=k)),no_of_class);
        test_data=zeros(length(Indices(Indices==k)),no_of_feature);
        test_target=zeros(length(Indices(Indices==k)),no_of_class);
        counter=1;
        counter2=1;
    for j = 1:no_of_sample
        if Indices(j)==k
            test_data(counter,:)=Sample(j,1:no_of_feature);
            test_target(counter,:)=Sample(j,no_of_feature+1:end);
            counter=counter+1;
        else
            train_data(counter2,:)=Sample(j,1:no_of_feature);
            train_target(counter2,:)=Sample(j,no_of_feature+1:end);
            counter2=counter2+1;
        end
    end
    train_target(train_target==0)=-1;
    test_target(test_target==0)=-1;
       train{i,1}=train_data;
       train{i,2}=train_target';
       train{i,3}=test_data;
       train{i,4}=test_target';
    end
    
end
function [A_train,b_train,A_test,b_test] = loadData
    spam_data = load('./spam/spambase.data');
    n = size(spam_data,1);
    test_size = ceil(n/5);
    test_indx = randsample(n,test_size,false);
    train_indx = setdiff([1:n]',test_indx);
    A_train = spam_data(train_indx,1:end-1);
    b_train = spam_data(train_indx,end);
    A_test = spam_data(test_indx,1:end-1);
    b_test = [spam_data(test_indx,end), 1-spam_data(test_indx,end)];
end
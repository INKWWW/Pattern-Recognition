%% ========PCA--95% & accuracy rate======== %%
clc;
close all;
clear all;

%% ========Loading the train data and test data======== %%
train_images = loadimage('train-images-idx3-ubyte');
train_labels = loadlabel('train-labels-idx1-ubyte');
test_images = loadimage('t10k-images-idx3-ubyte');
test_labels = loadlabel('t10k-labels-idx1-ubyte');

%% =========PCA Algorithm========== %%
mean_train_images = mean(train_images, 2);

%calculate the covariance matrix
X = train_images - mean_train_images;
S = X * X';
[eigenvector, eigenvalue] = eig(S);  %The columns of 'eigenvector' are eigenvectors of S, the diagonal value are eigenvalue of S
for n = 1:784
    new_eigenvalue(1, n) = eigenvalue(n, n);
end
[sort_eigenvalue, index1] = sort(new_eigenvalue, 'descend');

%% =========find p to satisfy 95%======== %%
threshold = 0.95;
denominator = sum(sort_eigenvalue);
numerator = 0;
for p = 1:784
     numerator = sort_eigenvalue(1, p) + numerator;
     if numerator / denominator >= threshold
         disp('p is: ')
         disp(p)
         break;
     end     
end
%% ======accuracy rate====== %%

%reduce the dimension to p
for m = 1 : p
    eigenvector_pd(:, m) = eigenvector(:, index1(1, m));
end
y_train_pd = (eigenvector_pd)' * train_images;
y_test_pd = (eigenvector_pd)' * test_images;

distance = zeros(10000, 60000);
%calculate the Euclidean distance
for column_test = 1:10000
    subtract_test_pd = repmat(y_test_pd(:, column_test), 1, 60000);
    dis_matrix = y_train_pd - subtract_test_pd;
    [dis, index]= min(sum(dis_matrix.^2));
    my_test_label(column_test, 1) = train_labels(index, 1);
end
%calculate the accuracy rate
error = 0;
for row = 1:10000
    if my_test_label(row, 1) ~= test_labels(row, 1)
        error = error + 1;
    end
end
acc_rate_pd = (10000 - error) / 10000;
disp('PCA_accuracy_rate_pd: ')
disp(acc_rate_pd)
%% ========PCA--80D--accuracy rate======== %%
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

%% ======80D====== %%

%reduce the dimension to 80D for train_images & test_images
for m = 1 : 80
    engienvector_80d(:, m) = eigenvector(:, index1(1, m));
end
y_train_80d = (engienvector_80d)' * train_images;
y_test_80d = (engienvector_80d)' * test_images;

%calculate the Euclidean distance for test_images
for column_test = 1:10000
    subtract_test_80d = repmat(y_test_80d(:, column_test), 1, 60000);
    dis_matrix = y_train_80d - subtract_test_80d;
    [dis, index]= min(sum(dis_matrix.^2));
    my_test_label(column_test, 1) = train_labels(index, 1);
end
%calculate the accuracy rate of test_images
error = 0;
for row = 1:10000
    if my_test_label(row, 1) ~= test_labels(row, 1)
        error = error + 1;
    end
end
acc_rate_80d_test = (10000 - error) / 10000;
disp('PCA_accuracy_rate_80d_test: ')
disp(acc_rate_80d_test)



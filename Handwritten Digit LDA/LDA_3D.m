%% ========LDA--3D======== %%
clc;
close all;
clear all;

%% ========Loading the train data and test data======== %%
train_images = loadimage('train-images-idx3-ubyte');
train_labels = loadlabel('train-labels-idx1-ubyte');
test_images = loadimage('t10k-images-idx3-ubyte');
test_labels = loadlabel('t10k-labels-idx1-ubyte');

%% =========LDA Algorithm========== %%
N = 60000;
%num of every classification 0~9
counter = zeros(10, 1);
for row = 1 : 60000
    for n = 0 : 9
        if train_labels(row, 1) == n
            counter(n + 1, 1) = counter(n + 1, 1) + 1;
        end
    end    
end
[sort_train_labels, index_train_labels] = sort(train_labels);
for column = 1 : 60000
    sort_train_images(:, column) = train_images(:, index_train_labels(column, 1));
end

%mean of specific class
mu_i = zeros(784, 10);
counter_sum = 0;
S_i = zeros(784, 7840);
for i = 0 : 9
    mid_sum = zeros(784, 1);
    for m = (1 + counter_sum) : (counter_sum + counter(i + 1, 1))
        mid_sum =  mid_sum + train_images(:, index_train_labels(m, 1));
    end
    mu_i(:, i + 1) = mid_sum / counter(i + 1, 1);
    
    %covariance of specific class    
    mu_i_rep = repmat(mu_i(:, i + 1), 1, counter(i + 1, 1));
    x_mui = sort_train_images(:, 1 + counter_sum : (counter_sum + counter(i + 1, 1))) - mu_i_rep;
    S_i(:, 784 * i + 1 : 784 * (i + 1)) = (x_mui * x_mui') / counter(i + 1, 1);    
    counter_sum = counter(i + 1, 1) + counter_sum;
end

%with-in class scatter
Sw = zeros(784, 784);
for i = 0 : 9
    Sw = Sw + (S_i(:, 784 * i + 1 : 784 * (i + 1))) * (counter(i + 1, 1) / N);
end

%total mean vector
mu = sum(train_images, 2) / N;

%between-class scatter
Sb = zeros(784, 784);
for i = 0 : 9
    mui_mu = mu_i(:, i + 1) - mu;
    Sb = Sb + (mui_mu * mui_mu') * (counter(i + 1, 1) / N);
end

%calculate the eigenvalue and eigenvector
  %make Sw to a full rank matrix
diagonal = eye(784) * 0.0001;
Sw_full_rank = Sw + diagonal;
  %get the eigen_matrix
eigen_matrix = Sw_full_rank^(-1) * Sb;
[eigenvector, eigenvalue] = eig(eigen_matrix);
for n = 1:784
    new_eigenvalue(1, n) = eigenvalue(n, n);
end
[sort_eigenvalue, index1] = sort(new_eigenvalue, 'descend');

%sort train_labels from 0 to 9
[sort_train_label, index2] = sort(train_labels);
for n = 1 : 60000
    sort_train_images(:, n) = train_images(:, index2(n, 1)); 
end
%% ======3D====== %%

%Projected value in 3D
engienvector_3d(:, 1) = eigenvector(:, index1(1, 1));
engienvector_3d(:, 2) = eigenvector(:, index1(1, 2));
engienvector_3d(:, 3) = eigenvector(:, index1(1, 3));
y_3d = (engienvector_3d)' * sort_train_images;

index_label = zeros(1, 10);
index_label(1, 1) = counter(1, 1);
for m = 2 : 10    
    index_label(1, m) = index_label(1, m - 1) + counter(m, 1);
end

%sketch these classified points
color = colormap(jet(10));
figure(1);
scatter3(y_3d(1, 1: index_label(1, 1)), y_3d(2, 1 : index_label(1, 1)), y_3d(3, 1 : index_label(1, 1)), 3, color(1, 1:3));
hold on;
for k = 2 : 10
    scatter3(y_3d(1, index_label(1, k - 1) : index_label(1, k)), y_3d(2, index_label(1, k - 1) : index_label(1, k)), y_3d(3, index_label(1, k - 1) : index_label(1, k)), 3, color(k, 1:3));
    hold on;
end
title('LDA-Visualize the projected data vector in 3d');

%sketch eigenvector
figure(2);
subplot(2, 2, 1);
engienvector1 = reshape(engienvector_3d(:, 1), 28, 28);
imshow(engienvector1, []);
title('LDA-Visualize the eigenvector1-3D');
subplot(2, 2, 2);
engienvector2 = reshape(engienvector_3d(:, 2), 28, 28);
imshow(engienvector2, []);
title('LDA-Visualize the eigenvector2-3D');
subplot(2, 2, 3);
engienvector3 = reshape(engienvector_3d(:, 3), 28, 28);
imshow(engienvector3, []);
title('LDA-Visualize the eigenvector3-3D');

%% ========calculate the accuracy rate of test_images ======== %%
%reduce the dimension to 3D for train_images & test_images
for m = 1 : 3
    engienvector_3d(:, m) = eigenvector(:, index1(1, m));
end
y_train_3d = (engienvector_3d)' * train_images;
y_test_3d = (engienvector_3d)' * test_images;

%calculate the Euclidean distance for test_images
for column_test = 1:10000
    subtract_test_3d = repmat(y_test_3d(:, column_test), 1, 60000);
    dis_matrix = y_train_3d - subtract_test_3d;
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
acc_rate_3d_test = (10000 - error) / 10000;
disp('LDA_accuracy_rate_3d_test: ')
disp(acc_rate_3d_test)
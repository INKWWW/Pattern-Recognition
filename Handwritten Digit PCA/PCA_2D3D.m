%% ========PCA--2D & 3D======== %%
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

%sort train_labels from 0 to 9
[sort_train_label, index2] = sort(train_labels);
for n = 1 : 60000
    sort_train_images(:, n) = train_images(:, index2(n, 1)); 
end
%% ======2D====== %%

%Projected value in 2D
engienvector_2d(:, 1) = eigenvector(:, index1(1, 1));
engienvector_2d(:, 2) = eigenvector(:, index1(1, 2));
y_2d = (engienvector_2d)' * sort_train_images;

%record num of 0 ~ 9
counter = zeros(1, 10);
for i = 0:9
    for j = 1:60000
        if sort_train_label(index2(j, 1), 1) == i
            counter(1, i + 1) = counter(1, i + 1) + 1;           
        end
    end
end
index_label = zeros(1, 10);
index_label(1, 1) = counter(1, 1);
for m = 2 : 10    
    index_label(1, m) = index_label(1, m - 1) + counter(1, m);
end

%sketch these classified points
color = colormap(jet(10));
figure(1);
% plot(engienvector_2d(:, 1), engienvector_2d(:, 2));
scatter(y_2d(1, 1: index_label(1, 1)), y_2d(2, 1 : index_label(1, 1)), 3, color(1, 1:3));
hold on;
for k = 2 : 10
    scatter(y_2d(1, index_label(1, k - 1) : index_label(1, k)), y_2d(2, index_label(1, k - 1) : index_label(1, k)), 3, color(k, 1:3));
    hold on;
end
title('PCA-Visualize the projected data vector in 2d');

%sketch eigenvector
figure(2);
subplot(1, 2, 1);
engienvector1 = reshape(engienvector_2d(:, 1), 28, 28);
imshow(engienvector1, []);
title('PCA-Visualize the eigenvector1-2D');
subplot(1, 2, 2);
engienvector2 = reshape(engienvector_2d(:, 2), 28, 28);
imshow(engienvector2, []);
title('PCA-Visualize the eigenvector2-2D');

%% ======3D====== %%

%Projected value in 3D
engienvector_3d(:, 1) = eigenvector(:, index1(1, 1));
engienvector_3d(:, 2) = eigenvector(:, index1(1, 2));
engienvector_3d(:, 3) = eigenvector(:, index1(1, 3));
y_3d = (engienvector_3d)' * sort_train_images;

%record num of 0 ~ 9
counter = zeros(1, 10);
for i = 0:9
    for j = 1:60000
        if sort_train_label(index2(j, 1), 1) == i
            counter(1, i + 1) = counter(1, i + 1) + 1;           
        end
    end
end
index_label = zeros(1, 10);
index_label(1, 1) = counter(1, 1);
for m = 2 : 10    
    index_label(1, m) = index_label(1, m - 1) + counter(1, m);
end

%sketch these classified points
color = colormap(jet(10));
figure(3);
scatter3(y_3d(1, 1: index_label(1, 1)), y_3d(2, 1 : index_label(1, 1)), y_3d(3, 1 : index_label(1, 1)), 3, color(1, 1:3));
hold on;
for k = 2 : 10
    scatter3(y_3d(1, index_label(1, k - 1) : index_label(1, k)), y_3d(2, index_label(1, k - 1) : index_label(1, k)), y_3d(3, index_label(1, k - 1) : index_label(1, k)), 3, color(k, 1:3));
    hold on;
end
title('PCA-Visualize the projected data vector in 3d');

%sketch eigenvector
figure(4);
subplot(2, 2, 1);
engienvector1 = reshape(engienvector_3d(:, 1), 28, 28);
imshow(engienvector1, []);
title('PCA-Visualize the eigenvector1-3D');
subplot(2, 2, 2);
engienvector2 = reshape(engienvector_3d(:, 2), 28, 28);
imshow(engienvector2, []);
title('PCA-Visualize the eigenvector2-3D');
subplot(2, 2, 3);
engienvector3 = reshape(engienvector_3d(:, 3), 28, 28);
imshow(engienvector3, []);
title('PCA-Visualize the eigenvector3-3D');

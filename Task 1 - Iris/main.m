clear all; 
clc;

%% Extract data
setosa_data         = load('Dataset/setosa.txt', '-ascii'); 
versicolor_data     = load('Dataset/versicolor.txt', '-ascii');
virginica_data      = load('Dataset/virginica.txt', '-ascii');

data(:, :, 1) = setosa_data;
data(:, :, 2) = versicolor_data;
data(:, :, 3) = virginica_data;

%% Setup & training
% Create W = [W~ w_0]
% z_k = W~ * x_k + w_0 = [W~ w_0] * [x_k' 1]'
% 
% Since number of features is 4, x_k = 4 x 1
%  
% W~    = number of classes x number of features    = 3 x 4
% w_0   = number of classes x 1                     = 3 x 1 
% W     = number of classes x 5                     = 3 x 5

num_classes = 3; 
num_features = 4; 
num_samples = size(setosa_data, 1);
num_training_samples = 30;
num_test_samples = num_samples - num_training_samples;

alpha = 0.01;
iteration = 0;
max_num_iterations = 10000;
tolerance = 0.01;

training_data = data(1:num_training_samples, :, :);
[W, MSE_values] = TrainClassifier(num_classes, num_features, max_num_iterations, tolerance, alpha, training_data);

figure;
plot(MSE_values);
title('MSE');
ylabel('MSE');
xlabel('Iterations');

%% Testing
test_data = data(num_training_samples + 1:num_samples, :, :);
confusion_matrix = TestClassifier(num_classes, W, test_data);
disp(confusion_matrix);
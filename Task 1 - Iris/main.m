clear all; 
clc;

%% Extract data
setosa_data         = load('Dataset/setosa.txt', '-ascii'); 
versicolor_data     = load('Dataset/versicolor.txt', '-ascii');
virginica_data      = load('Dataset/virginica.txt', '-ascii');

data(:, :, 1) = setosa_data;
data(:, :, 2) = versicolor_data;
data(:, :, 3) = virginica_data;

%% Setup

% Since number of features is 4, x_k = 4 x 1
%  
% W~    = number of classes x number of features    = 3 x 4
% w_0   = number of classes x 1                     = 3 x 1 
% W     = number of classes x 5                     = 3 x 5

parameters.num_samples = size(setosa_data, 1);
parameters.alpha = 0.01; 
parameters.tolerance = 0.01;
parameters.max_num_iterations = 10000;
parameters.num_classes = 3;
parameters.num_features = 4;
parameters.plot = false;


%% Using 30 samples for training and 20 for testing
num_training_samples = 30;
TrainAndTestClassifier(num_training_samples, parameters, data);

%% Using 20 samples for training and 30 for testing
num_training_samples = 20;
TrainAndTestClassifier(num_training_samples, parameters, data);
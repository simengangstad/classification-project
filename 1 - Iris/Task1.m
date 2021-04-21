clear all; 
clc;

%% Extract data
setosa_data         = load('Dataset/setosa.txt', '-ascii'); 
versicolor_data     = load('Dataset/versicolor.txt', '-ascii');
virginica_data      = load('Dataset/virginica.txt', '-ascii');

% Merge the data into a single three dimensional matrix
data(:, :, 1) = setosa_data;
data(:, :, 2) = versicolor_data;
data(:, :, 3) = virginica_data;

%% Setup
parameters.num_samples = size(data, 1);
parameters.num_classes = 3;                 % We're classifying Setosa,
                                            % Versicolor and Virginica

parameters.num_features = 4;                % Width and height of both
                                            % sepal and petal

parameters.alpha = 0.01;                    % Step coefficient in the 
                                            % gradient descent

parameters.tolerance = 0.01;                % Tolerance for mean square 
                                            % error
                                            
parameters.max_num_iterations = 20000;      % If tolerance is not met, the
                                            % training will finalize after
                                            % this amount of iterations
                                            
parameters.plot = true;


%% Using 30 samples for training and 20 for testing
num_training_samples = 30;
TrainAndTestClassifier(num_training_samples, parameters, data);

%% Using 20 samples for training and 30 for testing
num_training_samples = 20;
TrainAndTestClassifier(num_training_samples, parameters, data);
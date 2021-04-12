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


%% Plot histograms

num_features = 4;
num_classes = 3;

feature_names = ["Sepal length" "Sepal width" "Petal length" "Petal width"];

tiledlayout(2, 2);

for feature_index = 1:num_features
    nexttile;
    for class_index = 1:num_classes
        features = data(:, feature_index, class_index);
        histogram(features);
        hold on;
    end
    
    title(sprintf('Feature: %s', feature_names(feature_index)));
    legend('Setosa', 'Versicolor', 'Virginica');
    hold off;
end

% From these results we see that sepal width has the most overlap and this
% feature is thus removed from the classifier


%% Setup
num_training_samples = 30;

parameters.num_samples = size(data, 1);
parameters.num_classes = 3;                 % We're classifying Setosa,
                                            % Versicolor and Virginica

parameters.num_features = 3;                % Sepal length and petal length
                                            % and width

parameters.alpha = 0.01;                    % Step coefficient in the 
                                            % gradient descent

parameters.tolerance = 0.01;                % Tolerance for mean square 
                                            % error
                                            
parameters.max_num_iterations = 10000;      % If tolerance is not met, the
                                            % training will finalize after
                                            % this amount of iterations
                                            
parameters.plot = false;

%% Remove the sepal width from the data
data(:, 2, :) = [];
disp('With sepal length, petal length and petal width:')
TrainAndTestClassifier(num_training_samples, parameters, data);

%% Remove the sepal length from the data
disp('With petal length and petal width:')
parameters.num_features = 2;                
data(:, 1, :) = [];
TrainAndTestClassifier(num_training_samples, parameters, data);

%% Remove the petal width from the data
disp('With petal length:')
parameters.num_features = 1;
data(:, 2, :) = [];
TrainAndTestClassifier(num_training_samples, parameters, data);
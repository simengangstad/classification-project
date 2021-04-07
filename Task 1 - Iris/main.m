%% Extract data
setosa_data         = load('Dataset/setosa.txt', '-ascii'); 
versicolor_data     = load('Dataset/versicolor.txt', '-ascii');
virginica_data      = load('Dataset/virginica.txt', '-ascii');

training_data(:, :, 1) = setosa_data;
training_data(:, :, 2) = versicolor_data;
training_data(:, :, 3) = virginica_data;

%% Setup 

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
num_traning_samples = 30;
num_test_samples = num_samples - num_traning_samples;

alpha = 1;

% Todo: ask what this should be.
% Offset being zero. 
W = [ones(num_classes, num_features) zeros(num_classes, 1)];
z_k = zeros(num_classes, 1);

% Generate t_k value for the specific class
t = eye(3);


%% Training

% Todo: test if it is different when training one class first or training one 
% sample from each 
for c = 1:num_classes
    for k = 1:num_traning_samples
        x_k = training_data(k, :, c)';
        z_k = W * [x_k' 1]';
        g_k = 1./(1 + exp(-z_k));
        
        disp('1');
        % sum (g_k - t_k) * g_k * ()

        % W = W - 
        
    end
end


%% Test

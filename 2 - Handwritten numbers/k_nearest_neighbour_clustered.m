% Script implementing clustering of training samples to obtain a KNN classifier

%% Setup data
clear;
data = load('Dataset/data.mat');

training_labels = data.trainlab;
training_data = data.trainv;

test_labels = data.testlab;
test_data = data.testv;

confusion_matrix = zeros(10, 10);

% Classifier properties
K = 10; 

%% Clustering
M = 64;                                     % Number of clusters 
number_of_classes = 10;

cluster_data = zeros(M * number_of_classes, 784);

tic;
for class = 1:number_of_classes
    class_indices = find(training_labels == (class - 1));
    class_data = training_data(class_indices, :);

    [~, class_cluster_data] = kmeans(class_data, M);

    cluster_index = (class - 1) * M;
    cluster_data(cluster_index + 1:cluster_index + M, :) = class_cluster_data;
    fprintf('Finished with class %d/10\n', class);
end


%% Nearest neighbour classifier 

% Find the Euclidean distance between the test data and the 
distance_matrix = dist(test_data, cluster_data');

% Find the entries with the lowest distance
[min_values, min_indices] = mink(distance_matrix', K, 1);

for i = 1:size(test_data, 1)
    min_index = min_indices(:, i);
    
    classified_labels = [];

    for j = 1:size(min_index, 1)
        classified_labels = [classified_labels, floor((min_index(j) - 1) / 64) + 1];
    end

    % If there are equal amount of some classes in the neighour, we need to 
    % differentiate between them using the distance
    unique_classes = unique(classified_labels);
    bins = histc(classified_labels, unique_classes);
    modes = unique_classes(bins == max(bins));

    min_distance = -1;
    classified_label = -1;

    for j = 1:size(modes, 2)
        class = modes(1, j);
        distance = 0;

        for index = 1:size(min_index, 2)
           if class == floor((min_index(index) - 1) / 64)
            distance = distance + min_values(index);
           end
        end

        if distance < min_distance || min_distance == -1
            min_distance = distance;
            classified_label = class;
        end
    end
    
    % Make 1 indexed for confusion matrix
    correct_label = test_labels(i) + 1;

    confusion_matrix(correct_label, classified_label) = confusion_matrix(correct_label, classified_label) + 1;
end
toc;

% Compute class error rate
for class_index = 1:size(confusion_matrix, 1)
    total_test_samples = sum(confusion_matrix(class_index, :));
    error_rate = 100 * (1 - confusion_matrix(class_index, class_index) / total_test_samples);
    fprintf('Error rate for number %d is %.2f. Total test samples: %d\n', (class_index - 1), error_rate, total_test_samples);
end

% Compute total error rate
sum_without_diagonal = (sum(confusion_matrix, 'all') - sum(diag(confusion_matrix)));
error_rate = 100 * sum_without_diagonal / sum(confusion_matrix, 'all');
fprintf('Total error rate: %.2f\n', error_rate);

disp('Confusion matrix');
disp(confusion_matrix);
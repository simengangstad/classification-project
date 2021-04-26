
% Script implementing clustering of training samples to obtain a nearest neighbour classifier with good performance. 

%% Setup data
data = load('Dataset/data.mat');

training_labels = data.trainlab;
training_data = data.trainv;

test_labels = data.testlab;
test_data = data.testv;

size_test_data = size(test_data,1);

%Arrays for classification:
misclassified_images = [];
misclassified_labels = [];

classified_images = [];

%Confusion matrix:
confusion_matrix = zeros(10, 10);

%% Clustering
M = 64; %Number of clusters 
[cluster_index, cluster_data] = kmeans(training_data,M);


%% Nearest neighbour classifier 

% Find the Euclidean distance between the test data and the training data
distance_matrix = dist(cluster_data, test_data');

% Find the entries with the lowest distance
%min_indices contains indices mapping to cluster_data, however we need to
%map clusterdata to training_labels.
[~, min_indices] = min(distance_matrix);

%For loop mapping 10000 classified tests
for i = 1:size_test_data
    min_index = min_indices(1, i);

    % Make 1 indexed for confusion matrix
    correct_label = test_labels(i) + 1;

    % For mapping: 
    training_indices = find(cluster_index == min_index);
    classified_label = generate_index(training_indices, training_labels);
    
    
%     if (correct_label ~= classified_label)
%         misclassified_images = [misclassified_images i];
%         misclassified_labels = [misclassified_labels classified_label-1];
%     else
%         classified_images = [classified_images i];
%     end
    
    %Construct confusion matrix
    confusion_matrix(correct_label, classified_label) = confusion_matrix(correct_label, classified_label) + 1;
end

% Compute class error rate
for class_index = 1:size(confusion_matrix, 1)
    total_test_samples = sum(confusion_matrix(class_index, :));
    error_rate = 1 - confusion_matrix(class_index, class_index) / total_test_samples;
    fprintf('Error rate for number %d is %.2f. Total test samples: %d\n', (class_index - 1), error_rate * 100, total_test_samples);
end

% Compute total error rate
sum_without_diagonal = (sum(confusion_matrix, 'all') - sum(diag(confusion_matrix)));
error_rate = sum_without_diagonal / sum(confusion_matrix, 'all');
fprintf('Total error rate: %d\n', error_rate);

disp('Confusion matrix');
disp(confusion_matrix);


function classified_label = generate_index(indices,training_labels)
    
    %Generate array of indices. 
    lables = training_labels(indices);
    classified_label = mode(lables); 
    classified_label = classified_label +1;
   
end
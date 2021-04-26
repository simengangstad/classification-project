%% Setup data
data = load('Dataset/data.mat');

training_labels = data.trainlab;
training_data = data.trainv;

test_labels = data.testlab;
test_data = data.testv;


%% Iterative construction of distance matrix in chunks and confusion matrix

confusion_matrix = zeros(10, 10);

chunk_training_size = 6000;
chunk_test_size = 1000;

% Holds the correct labels and classified labels for all test images.
classification_table = zeros(size(test_data, 1), 2);

% Loop through the chunks 
for chunk_index = 1:size(training_labels, 1) / chunk_training_size 

    chunk_training_start = (chunk_index - 1) * chunk_training_size + 1;
    
    chunk_training_data     = training_data(chunk_training_start:chunk_training_start + chunk_training_size - 1, :);
    chunk_training_labels   = training_labels(chunk_training_start:chunk_training_start + chunk_training_size - 1, :);
    
    
    chunk_test_start = (chunk_index - 1) * chunk_test_size + 1;
    
    chunk_test_data         = test_data(chunk_test_start:chunk_test_start + chunk_test_size - 1, :);
    chunk_test_labels       = test_labels(chunk_test_start: chunk_test_start + chunk_test_size - 1, :);
    
    % Find the Euclidean distance between the test data and the training data
    distance_matrix = dist(chunk_test_data, chunk_training_data');

    % Find the entries with the lowest distance
    [~, min_indices] = min(distance_matrix');
    
    % Chunk_size: is the size of the current chunk
    % i : the iteration of the distance matrix
    for i = 1:chunk_test_size
        min_index = min_indices(1, i);

        % Make 1 indexed for confusion matrix
        correct_label = chunk_test_labels(i) + 1; 
        classified_label = chunk_training_labels(min_index) + 1;
        
        current_training_index = (chunk_index - 1) * chunk_test_size + i;
        classification_table(current_training_index, :) = [correct_label - 1 classified_label - 1];
        
        confusion_matrix(correct_label, classified_label) = confusion_matrix(correct_label, classified_label) + 1;
    end
    
    fprintf('Chunk iteration %d/%d\n', chunk_index, size(training_labels, 1) / chunk_training_size);
end

%% Print error rate and confusion matrix

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


%% Build indices of misclassified and correctly classified images

misclassified_images = [];
correctly_classified_images = [];

for i = 1:size(classification_table, 1)
    
    correct_label = classification_table(i, 1);
    classified_label = classification_table(i, 2);
    
    if correct_label ~= classified_label
        misclassified_images = [misclassified_images i];
    else
        correctly_classified_images = [correctly_classified_images i];
    end
end

%% Plot classified and missclassified pictures
close;

rows = 3;
tiledlayout(rows, 2);

for row = 1:rows
    index = randi([1 size(correctly_classified_images, 2)], 1, 1);
    nexttile;
    plot_image(correctly_classified_images(1, index), test_data, classification_table);

    index = randi([1 size(misclassified_images, 2)], 1, 1);
    nexttile;
    plot_image(misclassified_images(1, index), test_data, classification_table);
end

function plot_image(index, data, classification_table) 
    x = zeros(28, 28); 
    x(:) = data(index, :); 
    image(x');
    
    correct_label = classification_table(index, 1);
    classified_label = classification_table(index, 2);

    if correct_label == classified_label
        title(sprintf('Classified correctly, label: %d', correct_label));
    else 
        title(sprintf('Misclassified, correct label: %d, classified label: %d', correct_label, classified_label));
    end
end

%% Setup data
data = load('Dataset/data.mat');

training_labels = data.trainlab;
training_data = data.trainv;

test_labels = data.testlab;
test_data = data.testv;

K = 7;


%% Iterative construction of distance matrix in chunks and confusion matrix
confusion_matrix = zeros(10, 10);

chunk_training_size = 6000;
chunk_test_size = 1000;

% Holds the correct labels and classified labels for all test images.
classification_table = zeros(size(test_data, 1), 2);

% Loop through the chunks 
tic;
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
    [min_values, min_indices] = mink(distance_matrix', K, 1);

    for i = 1:size(chunk_test_data, 1)
        min_index = min_indices(:, i);
        
        classified_labels = chunk_training_labels(min_index);

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
                if class == chunk_training_labels(index)
                    distance = distance + min_values(index);
                end
            end

            if distance < min_distance || min_distance == -1
                min_distance = distance;
                classified_label = class;
            end
        end
        
        % Make 1 indexed for confusion matrix
        correct_label = chunk_test_labels(i) + 1;

        confusion_matrix(correct_label, classified_label + 1) = confusion_matrix(correct_label, classified_label + 1) + 1;
    end

    fprintf('Chunk iteration %d/%d\n', chunk_index, size(training_labels, 1) / chunk_training_size);
end
toc;

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


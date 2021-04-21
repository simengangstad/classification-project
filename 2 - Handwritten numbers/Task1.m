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

for chunk_index = 1:size(training_labels, 1) / chunk_training_size

    chunk_training_start = (chunk_index - 1) * chunk_training_size + 1;
    
    chunk_training_data     = training_data(chunk_training_start:chunk_training_start + chunk_training_size - 1, :);
    chunk_training_labels   = training_labels(chunk_training_start:chunk_training_start + chunk_training_size - 1, :);
    
    
    chunk_test_start = (chunk_index - 1) * chunk_test_size + 1;
    
    chunk_test_data         = test_data(chunk_test_start:chunk_test_start + chunk_test_size - 1, :);
    chunk_test_labels       = test_labels(chunk_test_start: chunk_test_start + chunk_test_size - 1, :);
    
    distance_matrix = dist(chunk_test_data, chunk_training_data');
    
    [~, min_indices] = min(distance_matrix');


    for i = 1:chunk_test_size
        min_index = min_indices(1, i);

        % Make 1 indexed for confusion matrix
        correct_label = chunk_test_labels(i) + 1; 
        classified_label = chunk_training_labels(min_index) + 1;

        confusion_matrix(correct_label, classified_label) = confusion_matrix(correct_label, classified_label) + 1;
    end
    
    fprintf('Chunk iteration %d/%d\n', chunk_index, size(training_labels, 1) / chunk_training_size);
end

disp(confusion_matrix);
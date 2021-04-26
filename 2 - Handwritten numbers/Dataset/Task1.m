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

classified_images = [];
misclassified_images = [];

% Array holding the corresponding class to missclassified test. 
misclassified_labels = []; 

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

        if (correct_label ~= classified_label) 
            misclassified_images = [misclassified_images current_training_index];
            misclassified_labels = [misclassified_labels classified_label-1];
        else 
            classified_images = [classified_images current_training_index];
        end 

        confusion_matrix(correct_label, classified_label) = confusion_matrix(correct_label, classified_label) + 1;
    end
    
    fprintf('Chunk iteration %d/%d\n', chunk_index, size(training_labels, 1) / chunk_training_size);
end

%% Print error rate and confusion matrix

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


%% Plot classified and missclassified pictures
tiledlayout(3, 2);

nexttile;
plot_image_vec(classified_images(1)',test_data,test_labels,misclassified_labels(1),1);
nexttile;
plot_image_vec(misclassified_images(1)',test_data,test_labels,misclassified_labels(1),0);

nexttile;
plot_image_vec(classified_images(2)',test_data,test_labels,misclassified_labels(2),1);
nexttile;
plot_image_vec(misclassified_images(2)',test_data,test_labels,misclassified_labels(2),0);

nexttile;
plot_image_vec(classified_images(3)',test_data,test_labels,misclassified_labels(3),1);
nexttile;
plot_image_vec(misclassified_images(3)',test_data,test_labels,misclassified_labels(3),0);

function plot_image_vec(index, test_data, test_label, classify_num, classified) 
    x = zeros(28,28); 
    x(:) = test_data(index,:); 
    image(x');
    
    correct_label = test_label(index);

    if classified
        title(sprintf('Classified correctly, label: %d', correct_label));
    else 
        title(sprintf('Misclassified, correct label: %d, classified label: %d', correct_label, classify_num));
    end
end

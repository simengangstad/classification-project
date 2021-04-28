function [confusion_matrix, classification_table] = KNearestNeighbours(data,...
                                                                       k,...
                                                                       amount_of_classes)
                                                 
    % Runs k-nearest neighbours algorithm on some data.
    %
    % Input:
    % 
    % data:                             Struct of test_data, test_labels,
    %                                   template_data and template_labels
    % 
    % k:                                Number of neighbours to check
    %                                   the test against
    % 
    % amount_of_classes:                The classes in the data set
    %
    %
    % Output:
    %
    % confusion_matrix                  Matrix where rows are the correct
    %                                   class and columns are the
    %                                   classified
    %
    % classification_table              A Nx2 matrix where N is the amount
    %                                   of test samples. Will have the
    %                                   correct class on the first column 
    %                                   and the classified class on the 
    %                                   second column for every test sample.
    
    test_data = data.test_data;
    test_labels = data.test_labels;
    
    template_data = data.template_data;
    template_labels = data.template_labels;
    
    confusion_matrix = zeros(amount_of_classes, amount_of_classes);
    
    classification_table = zeros(size(test_data, 1), 2);

    % Find the Euclidean distance between the test data and the 
    % training data
    distance_matrix = dist(test_data, template_data');

    % Find the k entries with the lowest distance
    [min_values, min_indices] = mink(distance_matrix', k, 1);

    for i = 1:size(test_data, 1)
        min_index = min_indices(:, i);
        
        classified_labels = template_labels(min_index);

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
                if class == template_labels(index)
                    distance = distance + min_values(index);
                end
            end

            if distance < min_distance || min_distance == -1
                min_distance = distance;
                
                % Make 1 indexed for confusion matrix
                classified_label = class + 1;
            end
        end
        
        % Make 1 indexed for confusion matrix
        correct_label = test_labels(i) + 1;
        confusion_matrix(correct_label, classified_label) =...
            confusion_matrix(correct_label, classified_label) + 1;

        % Make 0 indexed for classification table so that it corresponds
        % to the numbers 
        classification_table(i, :) = [correct_label-1 classified_label-1];
    end
end


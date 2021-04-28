function [confusion_matrix, classification_table] =... 
         ChunkedKNearestNeighbours(data,...
                                   k,...
                                   amount_of_classes,...
                                   chunk_template_size,...
                                   chunk_test_size)
    
                                                     
    % Runs k-nearest neighbours algorithm on some data which is processed 
    % in chunks.
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
                                                         
    for chunk_index = 1:size(template_labels, 1) / chunk_template_size 
        
        chunk_template_start = (chunk_index - 1) * chunk_template_size + 1;
        template_interval = chunk_template_start:chunk_template_start +...
                                        chunk_template_size - 1;
                                    
        chunk_template_data = template_data(template_interval, :);
        chunk_template_labels = template_labels(template_interval, :);


        chunk_test_start = (chunk_index - 1) * chunk_test_size + 1;
        test_interval = chunk_test_start:chunk_test_start +...
                                         chunk_test_size - 1;
                                     
        chunk_test_data = test_data(test_interval, :);
        chunk_test_labels = test_labels(test_interval, :);

        chunk_data.test_data = chunk_test_data;
        chunk_data.test_labels = chunk_test_labels;
        chunk_data.template_data = chunk_template_data;
        chunk_data.template_labels = chunk_template_labels;

        [confusion_matrix_chunk, classification_table_chunk] =...
            KNearestNeighbours(chunk_data, k, amount_of_classes);

        confusion_matrix = confusion_matrix + confusion_matrix_chunk;
        
        chunk_interval = chunk_test_start:chunk_test_start +...
            size(classification_table_chunk, 1) - 1;
        
        classification_table(chunk_interval, :) =...
            classification_table_chunk;

        fprintf('Chunk iteration %d/%d\n',...
            chunk_index, size(template_labels, 1) / chunk_template_size);
    end
end


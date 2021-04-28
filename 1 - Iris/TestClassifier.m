function [confusion_matrix, error_rate] = TestClassifier(num_classes,...
                                                         W,...
                                                         data)

    % TestClassifier Does gradient descent with a cost function in order
    %                 to find the weight and bias matrix W
    %
    % num_classes:                      Number of classes
    % W:                                The matrix consisting of the 
    %                                   weights and biases from the
    %                                   training
    % data:                             Data tested on classifier
    %
    %
    % Returns:
    % confusion_matrix                  Confusion matrix after the testing,
    %                                   which specifies how well the
    %                                   classifier has classified
    % error_rate                        Error rate of the classifier
    
    
    confusion_matrix = zeros(num_classes, num_classes);
    
    for c = 1:num_classes
        for k = 1:size(data, 1)
            
            % Run the classifier on the test data
            x_k = [data(k, :, c) 1]'; 
            z_k = W * x_k;

            % Sigmoid function
            g_k = 1./(1 + exp(-z_k));

            % Retrieve index (which corresponds to class) the classifier
            % thought this sample is
            [~, max_index] = max(g_k); 

            previous_value = confusion_matrix(c, max_index);
            confusion_matrix(c, max_index) = previous_value + 1;
        end 
    end

    sum_without_diagonal = (sum(confusion_matrix, 'all') -...
                            sum(diag(confusion_matrix)));
                        
    error_rate =  sum_without_diagonal / sum(confusion_matrix, 'all'); 
end


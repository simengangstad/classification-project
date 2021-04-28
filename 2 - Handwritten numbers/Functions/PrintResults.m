function PrintResults(confusion_matrix)
    % Prints the error rate for each class in the confusion matrix, the
    % total error rate and the confusion matrix itself

    % Compute class error rate
    for class_index = 1:size(confusion_matrix, 1)
        total_test_samples = sum(confusion_matrix(class_index, :));
        
        error_rate = (1 - confusion_matrix(class_index, class_index) /...
            total_test_samples) * 100;
        
        output = 'Error rate for class %d is %.2f. Total samples: %d\n';
        fprintf(output, (class_index - 1), error_rate, total_test_samples);
    end

    % Compute total error rate
    sum_without_diagonal = (sum(confusion_matrix, 'all') -...
        sum(diag(confusion_matrix)));
    
    error_rate = 100 * sum_without_diagonal / sum(confusion_matrix, 'all');
    fprintf('Total error rate: %.2f\n', error_rate);

    
    disp('Confusion matrix');
    disp(confusion_matrix);
end


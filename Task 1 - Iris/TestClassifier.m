function [confusion_matrix, error_rate] = TestClassifier(num_classes, W, data)

    confusion_matrix = zeros(num_classes, num_classes);
    
    for c = 1:num_classes
        for k = 1:size(data, 1)
            
            x_k = [data(k, :, c) 1]'; 
            z_k = W * x_k;

            % Sigmoid function
            g_k = 1./(1 + exp(-z_k));

            [~, max_index] = max(g_k); 

            confusion_matrix(c, max_index) = confusion_matrix(c, max_index) + 1;
        end 
    end

    error_rate = (sum(confusion_matrix, 'all') - sum(diag(confusion_matrix))) / sum(confusion_matrix, 'all'); 
end


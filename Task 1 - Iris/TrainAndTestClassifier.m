function TrainAndTestClassifier(num_training_samples, parameters, data)

    training_data = data(1:num_training_samples, :, :);
    [W, MSE_values] = TrainClassifier(parameters.num_classes, parameters.num_features, parameters.max_num_iterations, parameters.tolerance, parameters.alpha, training_data);

    if parameters.plot
        figure;
        plot(MSE_values);
        title('MSE');
        ylabel('MSE');
        xlabel('Iterations');
    end
    
    % Testing
    test_data = data(num_training_samples + 1:parameters.num_samples, :, :);
    [confusion_matrix, error_rate] = TestClassifier(parameters.num_classes, W, test_data);
    fprintf('Number of samples used for training: %d\n', num_training_samples);
    disp('Confusion matrix:');
    disp(confusion_matrix);

    disp('Error rate:');
    disp(error_rate);
end


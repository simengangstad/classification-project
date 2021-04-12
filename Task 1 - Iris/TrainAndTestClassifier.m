function TrainAndTestClassifier(num_training_samples, parameters, data)
    % TrainAndTestClassifier Is a convenience function for testing the
    %                        training and testing for different amount of
    %                        training samples and parameters
    %
    % parameters.num_samples:           Total number of samples for each
    %                                   class in the data
    % parameters.num_classes:           Number of classes
    % parameters.num_features:          Number of features for each class
    % parameters.alpha:                 Step coefficient in the gradient 
    %                                   descent
    % parameters.tolerance:             Tolerance for mean square error,
    %                                   when the training is satisfied
    % parameters.max_num_iterations:    IF the training reaches this number
    %                                   it will abort even if tolerance is
    %                                   not met
    % data                              Training data and test data

    training_data = data(1:num_training_samples, :, :);
    [W, MSE_values] = TrainClassifier(parameters.num_classes,... 
                                      parameters.num_features,...
                                      parameters.alpha,...
                                      parameters.tolerance,...
                                      parameters.max_num_iterations,... 
                                      training_data);

    if parameters.plot
        figure;
        plot(MSE_values);
        title('MSE');
        ylabel('MSE');
        xlabel('Iterations');
    end
    
    % Testing
    test_data = data(num_training_samples + 1:parameters.num_samples,... 
                     :,... 
                     :);
    
    [confusion_matrix, error_rate] = TestClassifier(...
                                        parameters.num_classes,...
                                        W,... 
                                        test_data);

                                                    
    fprintf(...
            'Number of samples used for training: %d\n',... 
            num_training_samples);
        
    disp('Confusion matrix:');
    disp(confusion_matrix);

    disp('Error rate:');
    disp(error_rate);
end


function [W, MSE_values] = TrainClassifier(num_classes,... 
                                           num_features,...
                                           alpha,...
                                           tolerance,... 
                                           max_num_iterations,... 
                                           data)
                                       
    % TrainClassifier Does gradient descent with a cost function in order
    %                 to find the weight and bias matrix W
    %
    % num_classes:                      Number of classes
    % num_features:                     Number of features for each class
    % alpha:                            Step coefficient in the gradient 
    %                                   descent
    % tolerance:                        Tolerance for mean square error,
    %                                   when the training is satisfied
    % max_num_iterations:               IF the training reaches this number
    %                                   it will abort even if tolerance is
    %                                   not met
    % data                              Training data
    %
    %
    % Returns:
    %
    % W:                                The matrix consisting of the 
    %                                   weights and biases
    % MSE_values:                       Vector of mean square error value
    %                                   for each iteration in the training.

    % Makes it possible to generate t_k vector for the specific class
    t = eye(num_classes);

    % Holds the maximum sqare error value for each iteration towards the 
    % converged value
    MSE_values = zeros(max_num_iterations, 1);

    
    % Since number of features is 4, x_k = 4 x 1
    %  
    % W~    = number of classes x number of features    = 3 x 4
    % w_0   = number of classes x 1                     = 3 x 1 
    % W     = number of classes x 5                     = 3 x 5
    
    % W = [W~ w_0]
    W = [zeros(num_classes, num_features) zeros(num_classes, 1)];
    
    iteration = 0;
    
    num_training_samples = size(data, 1);
    
    shouldIterate = true; 
    while shouldIterate
        MSE = 0;
        GMSE = 0;

        for c = 1:num_classes

            % Target vector for the given class, what we want to classify
            t_k = t(:, c);

            for k = 1:num_training_samples
                x_k = [data(k, :, c) 1]'; 

                % z_k = W~ * x_k + w_0 = [W~ w_0] * [x_k' 1]'
                z_k = W * x_k;

                % Sigmoid function
                g_k = 1./(1 + exp(-z_k));

                % Gradient minimum square error 
                GMSE = GMSE + ((g_k - t_k) .* g_k .* (1 - g_k)) * x_k';

                MSE = MSE + (g_k - t_k)' * (g_k - t_k);
            end
        end

        W = W - alpha * GMSE;

        iteration = iteration + 1;

        MSE = 0.5 * MSE;
        MSE_values(iteration, 1) = MSE;

        shouldIterate = abs(MSE) > tolerance && iteration < max_num_iterations;
    end
end


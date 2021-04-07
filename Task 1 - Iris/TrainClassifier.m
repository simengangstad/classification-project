function [W, MSE_values] = TrainClassifier(num_classes, num_features, max_num_iterations, tolerance, alpha, data)

    % Makes it possible to generate t_k vector for the specific class
    t = eye(num_classes);

    % Holds the maximum sqare error value for each iteration towards the 
    % converged value
    MSE_values = zeros(max_num_iterations, 1);

    W = [zeros(num_classes, num_features) zeros(num_classes, 1)];
    
    iteration = 0;
    
    num_training_samples = size(data, 1);
    
    shouldIterate = true; 
    while shouldIterate
        MSE = 0;
        GMSE = 0;

        for c = 1:num_classes

            % Target vector
            t_k = t(:, c);

            for k = 1:num_training_samples
                x_k = [data(k, :, c) 1]'; 
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


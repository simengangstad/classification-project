function data = ClusterData(rawdata,...
                            number_of_clusters,...
                            number_of_classes)
    
    % ClusterData clusters some raw data of N classes and M clusters
                        
    template_labels = rawdata.trainlab;
    template_data = rawdata.trainv;
    test_labels = rawdata.testlab;
    test_data = rawdata.testv;
                                           
    cluster_labels = [];
    cluster_data = zeros(number_of_clusters * number_of_classes, 784);

    for class = 1:number_of_classes
        class_indices = find(template_labels == (class - 1));
        class_data = template_data(class_indices, :);

        [~, class_cluster_data] = kmeans(class_data, number_of_clusters);

        cluster_index = (class - 1) * number_of_clusters;
        
        interval = cluster_index + 1:cluster_index + number_of_clusters;
        cluster_data(interval, :) = class_cluster_data;
        
        cluster_labels = [cluster_labels; (class - 1) *...
            ones(number_of_clusters, 1)];
        fprintf('Finished clustering class %d/10\n', class);
    end

    data.template_labels = cluster_labels;
    data.template_data = cluster_data;
    data.test_labels = test_labels;
    data.test_data = test_data;
end


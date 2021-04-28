function PlotRandomImages(classification_table, test_data)

    % Build indices of misclassified and correctly classified images
    misclassified_images = [];
    correctly_classified_images = [];

    for i = 1:size(classification_table, 1)

        correct_label = classification_table(i, 1);
        classified_label = classification_table(i, 2);

        if correct_label ~= classified_label
            misclassified_images = [misclassified_images i];
        else
            correctly_classified_images = [correctly_classified_images i];
        end
    end

    % Plot classified and missclassified pictures
    close;

    rows = 3;
    tiledlayout(rows, 2);

    for row = 1:rows
        index = randi([1 size(correctly_classified_images, 2)], 1, 1);
        nexttile;
        plotImage(correctly_classified_images(1, index), test_data, classification_table);

        index = randi([1 size(misclassified_images, 2)], 1, 1);
        nexttile;
        plotImage(misclassified_images(1, index), test_data, classification_table);
    end
end


function plotImage(index, data, classification_table) 
    x = zeros(28, 28); 
    x(:) = data(index, :); 
    image(x');

    correct_label = classification_table(index, 1);
    classified_label = classification_table(index, 2);

    if correct_label == classified_label
        title(sprintf('Classified correctly, label: %d', correct_label));
    else 
        title(sprintf('Misclassified, correct label: %d, classified label: %d', correct_label, classified_label));
    end
end


